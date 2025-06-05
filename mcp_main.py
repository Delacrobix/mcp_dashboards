import base64
import json
import logging  # TODO: remove this
import os
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv()


from pydantic import BaseModel


class ChartType(str, Enum):
    lnsMetric = "metric chart"
    lnsPie = "pie chart"
    lnsXY = "xy chart"


class ChartResultModel(BaseModel):
    chart_type: ChartType
    group_by_field: Optional[str]  # Main field for chart
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
    ) -> List["ChartResultModel"]:
        """Analyze Dashboard image and match fields using OpenAI Vision, returning both analysis and field mapping in one call."""

        chart_schema = json.dumps(ChartResultModel.model_json_schema(), indent=2)
        prompt = f"""
        You are an expert in analyzing Kibana dashboards from images for the version 9.0.0 of Kibana.

        You will be given a dashboard image and a Elasticsearch index mappings.

        Below is the index mappings for the index that the dashboard is based on.
        Use this to help you understand the data and the fields that are available.

        Index Mappings:
        {index_mappings}

        For each chart you detect, return a JSON object with the following structure (see JSON schema below):
        {chart_schema}
        
        There some things to consider:
        - group_by_field: List of fields that are relevant for the chart, this is the data that will be used to generate the chart and can be found in the index mappings.
        - If you choose one field have multi-field, you can use the field name with the suffix .keyword to get the multi-field. For example: 
        
        'field_name': {{
            'type': 'text',
            'fields': {{
                'keyword': {{
                    'type': 'keyword',
                }}
            }}
        }}
        Here you can use the field_name.keyword to get the multi-field.
        
        'field_name': {{
            'type': 'keyword',
        }}
        Here you can use the field_name to get the single field.
        
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

            # Validate and parse as list of ChartResultModel
            if isinstance(result, list):
                charts = [ChartResultModel(**item) for item in result]
            elif isinstance(result, dict) and "charts" in result:
                charts = [ChartResultModel(**item) for item in result["charts"]]
            else:
                charts = []

            logging.info(f"Charts: {charts}")

            return charts

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
        chart: "ChartResultModel",
    ) -> Dict[str, Any]:
        """Fill the Lens template with the analyzed data and matched fields (ChartResultModel)"""

        template_str = json.dumps(template)
        chart_type = (
            chart.chart_type.value
            if isinstance(chart.chart_type, Enum)
            else chart.chart_type
        )

        replacements = {}

        if chart_type == "pie chart":
            group_field = chart.group_by_field or "category.keyword"
            replacements.update(
                {
                    "{group_field}": group_field,
                }
            )
        elif chart_type == "xy chart":
            x_field = chart.group_by_field or "category.keyword"
            replacements.update(
                {
                    "{x_field}": x_field,
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

        charts = generator.analyze_image_and_match_fields(
            image_base64, json.dumps(mappings, indent=2)
        )

        if not charts or isinstance(charts, dict) and "error" in charts:
            return {"error": "No charts detected or error in analysis."}

        panels = []
        for chart in charts:
            template = generator.get_chart_template(
                chart.chart_type.value
                if isinstance(chart.chart_type, Enum)
                else chart.chart_type
            )
            filled_panel = generator.fill_template_with_analysis(template, chart)
            panels.append(filled_panel)

        dashboard_result = generator.create_kibana_dashboard(panels, dashboard_title)

        return dashboard_result["dashboard_url"]

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
