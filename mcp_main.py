import base64
import json
import logging
import os
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from PIL import Image as PILImage

# Load environment variables
load_dotenv()

# Configuration
CHART_TYPES = ["lnsMetric", "lnsPie", "lnsXY"]


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
        self.index_name = "vet-visits"

        # Load templates from JSON files
        self.templates = self._load_templates()
        # self.series_types = self._load_series_types()

    def _load_templates(self) -> Dict[str, Any]:
        """Load visualization templates from JSON files"""
        templates = {}
        template_dir = os.path.join(os.path.dirname(__file__), "templates")

        for chart_type in CHART_TYPES:
            template_file = os.path.join(template_dir, f"{chart_type}.json")
            try:
                with open(template_file, "r") as f:
                    templates[chart_type] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Template file {template_file} not found")
                templates[chart_type] = {}

        return templates

    # def _load_series_types(self) -> Dict[str, str]:
    #     """Load series type mappings from JSON file"""
    #     series_file = os.path.join(
    #         os.path.dirname(__file__), "templates", "series_types.json"
    #     )
    #     try:
    #         with open(series_file, "r") as f:
    #             return json.load(f)
    #     except FileNotFoundError:
    #         print("Warning: Series types file not found, using defaults")
    #         return {
    #             "line_chart": "line",
    #             "bar_chart": "bar_stacked",
    #             "area_chart": "area_stacked",
    #         }

    def analyze_image_with_llm(
        self, image_base64: str, index_mappings: str
    ) -> Dict[str, Any]:
        """Analyze chart image using OpenAI Vision to extract parameters"""

        prompt = f"""
        Analyze this chart image and extract the following information:
        
        1. Chart type (must be one of: {', '.join(CHART_TYPES)})
        2. Number of data series/columns visible
        3. Field names that appear to be analyzed (axis labels, legends, etc.)
        4. Approximate values or ranges shown
        5. Time range if it's a time-series chart
        6. Any filters or aggregations that seem to be applied
        
        Available chart types:
        - lnsMetric: Single metric/KPI displays
        - lnsPie: Pie charts and donut charts  
        - lnsXY: Line charts, bar charts, area charts, scatter plots
    
        Below is the index mappings for the index that the chart is based on.
        Use this to help you understand the data and the fields that are available.

        Index Mappings:
        {index_mappings}
        
        Return the analysis as a JSON object with the following structure:
        {{
            "chart_type": "lnsMetric|lnsPie|lnsXY",
            "data_fields": ["field1", "field2"],
            "metrics": ["metric1", "metric2"],
            "time_field": "timestamp_field_if_applicable",
            "filters": [],
            "aggregation_type": "terms|date_histogram|avg|sum|count|unique_count",
            "approximate_values": {{}},
            "description": "Brief description of what the chart shows"
        }}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
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
                max_tokens=1000,
            )

            # Parse the JSON response
            content = response.choices[0].message.content
            # Extract JSON from the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]

            return json.loads(json_str)

        except Exception as e:
            return {
                "error": f"Failed to analyze image: {str(e)}",
                "chart_type": "lnsXY",  # default fallback
                "series_type": "bar_chart",
                "data_fields": [],
                "metrics": [],
                "description": "Analysis failed",
            }

    def get_elasticsearch_mappings(self, index_name: str) -> Dict[str, Any]:
        """Get mappings from Elasticsearch index"""
        try:
            mappings = self.es_client.indices.get_mapping(index=index_name)
            return mappings[list(mappings.keys())[0]]["mappings"]["properties"]
        except Exception as e:
            return {"error": f"Failed to get mappings: {str(e)}"}

    def match_fields_with_mappings(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Match analyzed fields with actual Elasticsearch field mappings"""

        prompt = f"""
        Given the chart analysis, match the fields correctly for Lens visualization:
        
        Chart Analysis: {json.dumps(analysis, indent=2)}
        
        For Lens visualizations, we need specific field assignments:
        - For lnsMetric: metric field for count/sum/avg operations
        - For lnsPie: group_by field for segments and metric field for values
        - For lnsXY: x_axis field (categorical/time) and y_axis field (metric)
        
        Return a JSON object with matched fields:
        {{
            "matched_fields": {{
                "metric_field": "actual_es_field_for_metrics",
                "group_by_field": "actual_es_field_for_grouping",
                "time_field": "actual_timestamp_field_if_applicable",
                "x_axis_field": "field_for_x_axis",
                "y_axis_field": "field_for_y_axis"
            }},
            "field_types": {{
                "field_name": "keyword|text|date|long|double|etc"
            }},
            "operation_types": {{
                "metric_operation": "count|sum|avg|unique_count|max|min",
                "group_operation": "terms|date_histogram"
            }}
        }}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]

            return json.loads(json_str)

        except Exception as e:
            return {"error": f"Failed to match fields: {str(e)}"}

    def get_chart_template(self, chart_type: str) -> Dict[str, Any]:
        """Get the Lens template for the specified chart type"""
        return self.templates.get(chart_type, self.templates.get("lnsXY", {}))

    def fill_template_with_analysis(
        self,
        template: Dict[str, Any],
        analysis: Dict[str, Any],
        matched_fields: Dict[str, Any],
        index_pattern_id: str,
    ) -> Dict[str, Any]:
        """Fill the Lens template with the analyzed data and matched fields"""

        # Generate unique IDs for the visualization
        panel_id = str(uuid.uuid4())
        layer_id = str(uuid.uuid4())

        template_str = json.dumps(template)

        # Replace placeholders based on chart type
        chart_type = analysis.get("chart_type", "lnsXY")

        # Common replacements
        replacements = {
            "{panel_id}": panel_id,
            "{layer_id}": layer_id,
            "{title}": analysis.get("description", "Generated Chart"),
            "{index_pattern_id}": index_pattern_id,
        }

        if chart_type == "lnsMetric":
            metric_accessor_id = str(uuid.uuid4())
            replacements.update(
                {
                    "{metric_accessor_id}": metric_accessor_id,
                    "{metric_label}": f"Count of {matched_fields.get('matched_fields', {}).get('metric_field', 'records')}",
                    "{operation_type}": matched_fields.get("operation_types", {}).get(
                        "metric_operation", "count"
                    ),
                    "{source_field}": matched_fields.get("matched_fields", {}).get(
                        "metric_field", "___records___"
                    ),
                }
            )

        elif chart_type == "lnsPie":
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
                    "{size}": "5",
                    "{metric_label}": "Count of records",
                    "{operation_type}": matched_fields.get("operation_types", {}).get(
                        "metric_operation", "count"
                    ),
                    "{source_field}": matched_fields.get("matched_fields", {}).get(
                        "metric_field", "___records___"
                    ),
                }
            )

        elif chart_type == "lnsXY":
            x_accessor_id = str(uuid.uuid4())
            y_accessor_id = str(uuid.uuid4())
            series_type = analysis.get("series_type", "bar_chart")
            lens_series_type = self.series_types.get(series_type, "bar_stacked")

            x_field = matched_fields.get("matched_fields", {}).get(
                "x_axis_field", "category.keyword"
            )
            y_field = matched_fields.get("matched_fields", {}).get(
                "y_axis_field", "___records___"
            )

            replacements.update(
                {
                    "{x_accessor_id}": x_accessor_id,
                    "{y_accessor_id}": y_accessor_id,
                    "{series_type}": lens_series_type,
                    "{x_label}": f"Top 10 values of {x_field}",
                    "{x_data_type}": "string",
                    "{x_operation_type}": "terms",
                    "{x_field}": x_field,
                    "{y_label}": "Count of records",
                    "{y_operation_type}": matched_fields.get("operation_types", {}).get(
                        "metric_operation", "count"
                    ),
                    "{y_field}": y_field,
                    "{size}": "10",
                }
            )

        # Apply all replacements
        for placeholder, value in replacements.items():
            template_str = template_str.replace(placeholder, str(value))

        return json.loads(template_str)

    def create_kibana_dashboard(
        self, panel_config: Dict[str, Any], dashboard_title: str
    ) -> Dict[str, Any]:
        """Create dashboard in Kibana using the Lens panel configuration"""

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
                    "panels": [panel_config],
                    # "tags": [],
                },
                # "options": {
                #     "hidePanelTitles": False,
                #     "useMargins": True,
                #     "syncColors": False,
                #     "syncCursor": True,
                #     "syncTooltips": True,
                # },
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

            dashboard_data = dashboard_response.json()
            dashboard_url = f"{self.kibana_url}/app/dashboards#/view/{dashboard_id}"

            return {
                "success": True,
                "dashboard_id": dashboard_id,
                "dashboard_url": dashboard_url,
            }

        except Exception as e:
            return {"error": f"Failed to create dashboard: {str(e)}"}

    def get_dashboard_screenshot(self, dashboard_id: str) -> Optional[str]:
        """Get a screenshot of the created dashboard using Kibana reporting API"""
        try:
            headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}

            if self.es_api_key:
                headers["Authorization"] = f"ApiKey {self.es_api_key}"

            # generate screenshot
            screenshot_response = requests.post(
                f"{self.kibana_url}/api/reporting/generate/png",
                headers=headers,
                json={
                    "layout": {
                        "id": "png",
                        "dimensions": {"width": 1950, "height": 1200},
                    },
                    "objectType": "dashboard",
                    "objectId": dashboard_id,
                },
            )

            if screenshot_response.status_code == 200:
                job_info = screenshot_response.json()
                job_id = job_info.get("job", {}).get("id")

                if job_id:

                    for _ in range(30):  # Wait up to 30 seconds
                        time.sleep(1)
                        status_response = requests.get(
                            f"{self.kibana_url}/api/reporting/jobs/download/{job_id}",
                            headers=headers,
                        )
                        if status_response.status_code == 200:
                            return base64.b64encode(status_response.content).decode(
                                "utf-8"
                            )

            return None

        except Exception as e:
            print(f"Failed to get screenshot: {str(e)}")
            return None


# Initialize the generator class
generator = KibanaDashboardGenerator()

# Initialize the MCP server
mcp = FastMCP("Kibana Dashboards MCP Server")


# TODO: Delete this tool, its just for testing
@mcp.tool()
def ping():
    """Ping the server to check if it is running"""
    return "pong"


# TODO: Delete this tool, its just for testing
@mcp.tool()
def create_thumbnail(image_path: str):
    """Create a thumbnail from an image and return as base64 string"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


@mcp.tool()
def analyze_chart_and_create_dashboard(
    image_path: str = "/Users/jeffreyrengifo/software/Freelance/2025-06-02_dashboards-llm-generator-article/utils/image copy.png",
    index_pattern_id: str = "68c2e548-5742-44c6-8ff2-1f18ae9cc3df",  # TODO: This should be dinamically generated
    dashboard_title: str = "Generated Dashboard",
) -> Dict[str, Any]:
    """
    Main function that analyzes a chart image and creates a Kibana dashboard using Lens

    Args:
        image_base64: Base64 encoded image of the chart
        index_pattern_id: Kibana index pattern ID to use for the dashboard
        dashboard_title: Title for the created dashboard

    Returns:
        Dictionary with dashboard creation results, screenshot, and URL
    """
    try:
        logging.info("Encoding image to base64...")
        image_base64 = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

        logging.info("Getting Elasticsearch mappings...")
        mappings = generator.get_elasticsearch_mappings(generator.index_name)

        logging.info("Analyzing image with LLM...")
        analysis = generator.analyze_image_with_llm(
            image_base64, json.dumps(mappings, indent=2)
        )

        if "error" in analysis:
            return {"error": f"Image analysis failed: {analysis['error']}"}

        if "error" in mappings:
            return {"error": f"Failed to get ES mappings: {mappings['error']}"}

        logging.info(f"Matching fields with mappings with analysis: {analysis} ...")
        matched_fields = generator.match_fields_with_mappings(analysis)

        if "error" in matched_fields:
            return {"error": f"Field matching failed: {matched_fields['error']}"}

        template = generator.get_chart_template(analysis.get("chart_type"))

        logging.info(f"Filling template with matching files {matched_fields} ...")
        filled_panel = generator.fill_template_with_analysis(
            template, analysis, matched_fields, index_pattern_id
        )

        # Save the filled panel to a file to be used for debugging
        visualization_id = str(uuid.uuid4())

        file_name = f"tmp/visualizations_generated/filled_panel_{visualization_id}.json"
        logging.info(f"Saving filled panel to {file_name} ...")
        with open(file_name, "w") as f:
            json.dump(filled_panel, f, indent=2)

        # filled_panel = {
        #     "type": "lens",
        #     "gridData": {
        #         "x": 0,
        #         "y": 0,
        #         "w": 12,
        #         "h": 12,
        #         "i": "0f24e42e-45b2-4bb2-b218-9aec5a1ebe00",
        #     },
        #     "panelIndex": "0f24e42e-45b2-4bb2-b218-9aec5a1ebe00",
        #     "panelConfig": {
        #         "attributes": {
        #             "title": "This metric displays the total number of visits, which is recorded as 3,014.",
        #             "visualizationType": "lnsMetric",
        #             "type": "lens",
        #             "references": [
        #                 {
        #                     "type": "index-pattern",
        #                     "id": "68c2e548-5742-44c6-8ff2-1f18ae9cc3df",
        #                     "name": "indexpattern-datasource-layer-db68ca35-6a35-4619-ac5b-86eeee7fc044",
        #                 }
        #             ],
        #             "state": {
        #                 "visualization": {
        #                     "layerId": "db68ca35-6a35-4619-ac5b-86eeee7fc044",
        #                     "layerType": "data",
        #                     "metricAccessor": "39202f36-3c33-415b-b5da-4076307796de",
        #                 },
        #                 "query": {"query": "", "language": "kuery"},
        #                 "filters": [],
        #                 "datasourceStates": {
        #                     "formBased": {
        #                         "layers": {
        #                             "db68ca35-6a35-4619-ac5b-86eeee7fc044": {
        #                                 "columns": {
        #                                     "39202f36-3c33-415b-b5da-4076307796de": {
        #                                         "label": "Count of Visits",
        #                                         "dataType": "number",
        #                                         "operationType": "count",
        #                                         "isBucketed": False,
        #                                         "scale": "ratio",
        #                                         "sourceField": "___records___",
        #                                         "params": {"emptyAsNull": True},
        #                                         "customLabel": True,
        #                                     }
        #                                 },
        #                                 "columnOrder": [
        #                                     "39202f36-3c33-415b-b5da-4076307796de"
        #                                 ],
        #                                 "incompleteColumns": {},
        #                                 "sampling": 1,
        #                                 "indexPatternId": "68c2e548-5742-44c6-8ff2-1f18ae9cc3df",
        #                             }
        #                         },
        #                         "currentIndexPatternId": "68c2e548-5742-44c6-8ff2-1f18ae9cc3df",
        #                     },
        #                     "indexpattern": {"layers": {}},
        #                     "textBased": {"layers": {}},
        #                 },
        #                 "internalReferences": [],
        #                 "adHocDataViews": {},
        #             },
        #             "enhancements": {},
        #         }
        #     },
        # }

        if "error" in filled_panel:
            return {"error": f"Template filling failed: {filled_panel['error']}"}

        logging.info("Creating dashboard with filled panel ...")
        dashboard_result = generator.create_kibana_dashboard(
            filled_panel, dashboard_title
        )

        if "error" in dashboard_result:
            return {"error": f"Dashboard creation failed: {dashboard_result['error']}"}

        logging.info("Getting dashboard screenshot ...")
        screenshot_base64 = generator.get_dashboard_screenshot(
            dashboard_result["dashboard_id"]
        )

        return {
            "success": True,
            "analysis": analysis,
            "matched_fields": matched_fields,
            "dashboard_url": dashboard_result["dashboard_url"],
            "dashboard_id": dashboard_result["dashboard_id"],
            "screenshot_base64": screenshot_base64,
            "message": f"Dashboard '{dashboard_title}' created successfully!",
        }

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
def get_available_indices() -> List[str]:
    """Get list of available Elasticsearch indices"""
    try:
        indices = generator.es_client.indices.get_alias(index="*")
        return list(indices.keys())
    except Exception as e:
        return [f"Error getting indices: {str(e)}"]


@mcp.tool()
def get_index_mappings(index_name: str) -> str:
    """Get detailed mappings for a specific index"""

    mappings = generator.get_elasticsearch_mappings(index_name)
    return json.dumps(mappings, indent=2)


@mcp.tool()
def get_available_index_patterns() -> List[Dict[str, str]]:
    """Get list of available Kibana index patterns with their IDs"""
    try:
        headers = {"Content-Type": "application/json", "kbn-xsrf": "true"}

        if generator.es_api_key:
            headers["Authorization"] = f"ApiKey {generator.es_api_key}"

        response = requests.get(
            f"{generator.kibana_url}/api/saved_objects/?typ_finde=index-pattern",
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            patterns = []

            for obj in data.get("saved_objects", []):
                patterns.append(
                    {
                        "id": obj["id"],
                        "title": obj["attributes"]["title"],
                        "name": obj["attributes"].get(
                            "name", obj["attributes"]["title"]
                        ),
                    }
                )

            return patterns
        else:
            return [{"error": f"Failed to get index patterns: {response.text}"}]

    except Exception as e:
        return [{"error": f"Error getting index patterns: {str(e)}"}]


if __name__ == "__main__":

    mcp.run()
