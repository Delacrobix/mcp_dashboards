import base64
import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from pydantic import TypeAdapter

load_dotenv()


# define data with pydantic
from pydantic import BaseModel, Field


class ChartType(str, Enum):
    lnsMetric = "metric chart"
    lnsPie = "pie chart"
    lnsXY = "xy chart"


class TemplateFields(BaseModel):
    panel_id: str = Field(description="The ID of the panel")
    layer_id: str = Field(description="The ID of the layer")
    title: str = Field(description="The title of the chart")
    metric_accessor_id: str = Field(description="The ID of the metric accessor")
    group_accessor_id: str = Field(description="The ID of the group accessor")


# TODO: Analyze if there are more fields that can be removed from the prompt
class AnalysisModel(BaseModel):
    chart_type: ChartType
    data_fields: List[str]
    description: str


class MatchedFieldsModel(BaseModel):
    metric_field: Optional[str]
    group_by_field: Optional[str]
    time_field: Optional[str]
    x_axis_field: Optional[str]
    y_axis_field: Optional[str]


class OperationTypesModel(BaseModel):
    metric_operation: Optional[str]
    group_operation: Optional[str]


class ChartResultModel(BaseModel):
    analysis: AnalysisModel
    matched_fields: MatchedFieldsModel
    operation_types: OperationTypesModel
    number_of_charts: int


class KibanaDashboardGenerator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.langchain_llm = ChatOpenAI(
            model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
        )
        self.es_client = Elasticsearch(
            [os.getenv("ELASTICSEARCH_URL")],
            api_key=os.getenv("ELASTICSEARCH_API_KEY"),
        )
        self.kibana_url = os.getenv("KIBANA_URL")
        self.es_api_key = os.getenv("ELASTICSEARCH_API_KEY")
        self.index_name = "kibana_sample_data_logs"

        # Load templates from JSON files
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """Load visualization templates from JSON files"""
        templates = {}
        template_dir = os.path.join(os.path.dirname(__file__), "templates")

        for chart_type in ChartType:
            match chart_type:
                case ChartType.lnsMetric:
                    template_file = os.path.join(template_dir, "lnsMetric.json")
                case ChartType.lnsPie:
                    template_file = os.path.join(template_dir, "lnsPie.json")
                case ChartType.lnsXY:
                    template_file = os.path.join(template_dir, "lnsXY.json")
                case _:
                    raise ValueError(f"Invalid chart type: {chart_type}")

            try:
                with open(template_file, "r") as f:
                    templates[chart_type.value] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Template file {template_file} not found")
                templates[chart_type.value] = {}

        return templates

    def analyze_image_and_match_fields(
        self, image_base64: str, index_mappings: str
    ) -> Dict[str, Any]:
        """Analyze Dashboard image and match fields using OpenAI Vision, returning both analysis and field mapping in one call."""

        # TODO: change the operation types to be more specific
        chart_schema = json.dumps(ChartResultModel.model_json_schema(), indent=2)
        prompt = f"""
        You are an expert in analyzing Kibana dashboards from images.

        For each chart you detect, return a JSON object with the following structure (see JSON schema below):

        {chart_schema}
        
        Below is the index mappings for the index that the dashboard is based on.
        Use this to help you understand the data and the fields that are available.

        Index Mappings:
        {index_mappings}

        Only include the fields that are relevant for each chart, based on what is visible in the image. The output must be a JSON array, with one object per chart.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return result

        except Exception as e:
            return {"error": f"Failed to analyze image and match fields: {str(e)}"}

    def get_elasticsearch_mappings(self, index_name: str) -> Dict[str, Any]:
        """Get mappings from Elasticsearch index"""
        try:
            mappings = self.es_client.indices.get_mapping(index=index_name)
            return mappings[list(mappings.keys())[0]]["mappings"]["properties"]
        except Exception as e:
            return {"error": f"Failed to get mappings: {str(e)}"}

    def get_chart_template(self, chart_type: str) -> Dict[str, Any]:
        """Get the Lens template for the specified chart type"""
        return self.templates.get(chart_type, self.templates.get("xy chart", {}))

    def fill_template_with_analysis(
        self,
        template: Dict[str, Any],
        analysis: Dict[str, Any],
        matched_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fill the Lens template with the analyzed data and matched fields"""

        logging.info(f"Filling template with analysis: {analysis}")

        # Generate unique IDs for the visualization
        panel_id = str(uuid.uuid4())
        layer_id = str(uuid.uuid4())

        template_str = json.dumps(template)

        chart_type = analysis.get("chart_type", "lnsXY")

        replacements = {
            "{panel_id}": panel_id,
            "{layer_id}": layer_id,
            "{title}": analysis.get("description", "Generated Chart"),
        }

        if chart_type == "metric chart":
            column_id = str(uuid.uuid4())
            replacements.update(
                {
                    "{column_id}": column_id,
                }
            )

        elif chart_type == "pie chart":
            group_accessor_id = str(uuid.uuid4())
            metric_accessor_id = str(uuid.uuid4())
            group_field = matched_fields.get("matched_fields", {}).get(
                "group_by_field", "category.keyword"
            )

            replacements.update(
                {
                    "{group_accessor_id}": group_accessor_id,
                    "{metric_accessor_id}": metric_accessor_id,
                    "{group_field}": group_field,
                    # "{size}": "5",
                    "{source_field}": matched_fields.get("matched_fields", {}).get(
                        "metric_field", "___records___"
                    ),
                }
            )

        elif chart_type == "xy chart":
            x_accessor_id = str(uuid.uuid4())
            y_accessor_id = str(uuid.uuid4())

            x_field = matched_fields.get("matched_fields", {}).get(
                "x_axis_field", "category.keyword"
            )

            # TODO: reduce to the minimun necessary fields
            replacements.update(
                {
                    "{x_accessor_id}": x_accessor_id,
                    "{y_accessor_id}": y_accessor_id,
                    "{x_label}": f"Top 10 values of {x_field}",
                    "{x_data_type}": "string",
                    "{x_operation_type}": "terms",
                    "{x_field}": x_field,
                    "{y_label}": "Count of records",
                    "{y_operation_type}": matched_fields.get("operation_types", {}).get(
                        "metric_operation", "count"
                    ),
                    "{size}": "10",
                }
            )

        for placeholder, value in replacements.items():
            template_str = template_str.replace(placeholder, str(value))

        return json.loads(template_str)

    def create_kibana_dashboard(
        self, panels: list, dashboard_title: str
    ) -> Dict[str, Any]:
        """Create dashboard in Kibana using the Lens panel configuration(s)"""

        try:
            headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}

            if self.es_api_key:
                headers["Authorization"] = f"ApiKey {self.es_api_key}"

            dashboard_id = str(uuid.uuid4())
            url = f"{self.kibana_url}/api/dashboards/dashboard/{dashboard_id}"

            dashboard_config = {
                "attributes": {
                    "title": dashboard_title,
                    "description": "Generated by MCP",
                    "timeRestore": False,
                    "panels": panels,
                    "timeFrom": "now-7d/d",
                    "timeTo": "now",
                },
            }

            dashboard_response = requests.post(
                url,
                headers=headers,
                json=dashboard_config,
            )

            if dashboard_response.status_code not in [200, 201]:
                return {
                    "error": f"Failed to create dashboard: {dashboard_response.text}"
                }

            dashboard_url = f"{self.kibana_url}/app/dashboards#/view/{dashboard_id}"

            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "dashboard_url": dashboard_url,
            }

        except Exception as e:
            return {"error": f"Failed to create dashboard: {str(e)}"}

    # TODO: Implement screenshot function properly
    def get_dashboard_screenshot(self, dashboard_id: str) -> Optional[str]:
        """Get a screenshot of the created dashboard using Kibana reporting API"""
        try:
            headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}

            if self.es_api_key:
                headers["Authorization"] = f"ApiKey {self.es_api_key}"

            # TODO: Generate screenshot properly
            screenshot_response = requests.post(
                f"{self.kibana_url}/api/reporting/generate/pngV2?jobParams=%28browserTimezone%3AAmerica%2FBogota%2Clayout%3A%28dimensions%3A%28height%3A344%2Cwidth%3A2158.400146484375%29%2Cid%3Apreserve_layout%29%2ClocatorParams%3A%28id%3ADASHBOARD_APP_LOCATOR%2Cparams%3A%28dashboardId%3A%2743c0dd5a-5a7e-40d6-a15b-34e11c016537%27%2Cpanels%3A%21%28%28gridData%3A%28h%3A12%2Ci%3A%274b404301-f252-4ad4-810e-ada08857cb99%27%2Cw%3A12%2Cx%3A0%2Cy%3A0%29%2CpanelConfig%3A%28attributes%3A%28description%3A%21n%2Cenhancements%3A%28%29%2Creferences%3A%21%28%28id%3A%2768c2e548-5742-44c6-8ff2-1f18ae9cc3df%27%2Cname%3Aindexpattern-datasource-layer-7c6767f8-a0b9-47a0-878b-3dd7ec6e9832%2Ctype%3Aindex-pattern%29%29%2CsavedObjectId%3A%21n%2Cstate%3A%28adHocDataViews%3A%28%29%2CdatasourceStates%3A%28formBased%3A%28layers%3A%28%277c6767f8-a0b9-47a0-878b-3dd7ec6e9832%27%3A%28columnOrder%3A%21%28a277cded-399b-40fd-99f1-afc46b3e06ff%29%2Ccolumns%3A%28a277cded-399b-40fd-99f1-afc46b3e06ff%3A%28customLabel%3A%21t%2CdataType%3Anumber%2Cfilter%3A%21n%2CisBucketed%3A%21f%2Clabel%3A%27Count%20of%20Visits%27%2CoperationType%3Acount%2Cparams%3A%28emptyAsNull%3A%21t%29%2CreducedTimeRange%3A%21n%2Cscale%3Aratio%2CsourceField%3A___records___%2CtimeScale%3A%21n%2CtimeShift%3A%21n%29%29%2CincompleteColumns%3A%28%29%2Csampling%3A1%29%29%29%29%2Cfilters%3A%21%28%29%2CinternalReferences%3A%21%28%29%2Cquery%3A%28language%3Akuery%2Cquery%3A%27%27%29%2Cvisualization%3A%28layerId%3A%277c6767f8-a0b9-47a0-878b-3dd7ec6e9832%27%2ClayerType%3Adata%2CmetricAccessor%3Aa277cded-399b-40fd-99f1-afc46b3e06ff%29%29%2Ctitle%3A%27This%20chart%20displays%20the%20total%20number%20of%20visits%2C%20quantified%20as%203%2C014.%27%2Ctype%3Alens%2CvisualizationType%3AlnsMetric%29%29%2CpanelIndex%3A%274b404301-f252-4ad4-810e-ada08857cb99%27%2Ctype%3Alens%29%29%2CpreserveSavedFilters%3A%21t%2CtimeRange%3A%28from%3Anow-1y%2Fd%2Cto%3Anow%29%2CuseHash%3A%21f%2CviewMode%3Aview%29%29%2CobjectType%3Adashboard%2Ctitle%3A%27Generated%20Dashboard%27%2Cversion%3A%279.0.0%27%29",
                headers=headers,
            )

            if screenshot_response.status_code == 200:
                return json.loads(screenshot_response)

            return None

        except Exception as e:
            print(f"Failed to get screenshot: {str(e)}")
            return None


generator = KibanaDashboardGenerator()
mcp = FastMCP("Kibana Dashboards MCP Server")


@mcp.tool()
def analyze_chart_and_create_dashboard(
    image_path: str = "/Users/jeffreyrengifo/software/Freelance/2025-06-02_dashboards-llm-generator-article/utils/dashboard.png",
    dashboard_title: str = "Generated Dashboard",
) -> Dict[str, Any]:
    """
    Main function that analyzes a chart image and creates a Kibana dashboard using Lens

    Args:
        image_path: Path to the chart image
        dashboard_title: Title for the created dashboard

    Returns:
        Dictionary with dashboard creation results, screenshot, and URL
    """
    try:
        image_base64 = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

        mappings = generator.get_elasticsearch_mappings(generator.index_name)

        result = generator.analyze_image_and_match_fields(
            image_base64, json.dumps(mappings, indent=2)
        )

        charts = result if isinstance(result, list) else result.get("charts", [])

        panels = []
        for chart in charts:

            analysis = chart.get("analysis", {})
            template = generator.get_chart_template(analysis.get("chart_type"))

            filled_panel = generator.fill_template_with_analysis(
                template, analysis, chart
            )

            # FIXME: This is for debugging purposes will be removed later
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filled_panel_file_name = f"tmp/visualizations_generated/{analysis.get('chart_type')}_{timestamp}.json"
            with open(filled_panel_file_name, "w") as f:
                json.dump(filled_panel, f, indent=2)

            panels.append(filled_panel)

        dashboard_result = generator.create_kibana_dashboard(panels, dashboard_title)

        # screenshot_base64 = generator.get_dashboard_screenshot(
        #     dashboard_result["dashboard_id"]
        # )

        return dashboard_result["dashboard_url"]

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
