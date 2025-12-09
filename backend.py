import json
from typing import Dict, Union, List, Any

import altair as alt
import pandas as pd
from IPython.display import display

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


Paper_field_inst = pd.read_csv('/Users/jialuli/Desktop/Yeshiva University Chatbot/backend/yeshiva_subset/Paper_field_inst_yu.csv')


params = {
    "institution": "Yeshiva University",     
    "field": "computer science",            
    "year_min": "2010",                        
    "year_max": "2024"                         
}

# Set defaults for the search query
def apply_defaults(params: Dict[str, Any]) -> Dict[str, Any]:

    institution = params.get("institution")
    # If institution is missing/empty, default to Yeshiva University
    if institution is None or str(institution).strip() == "":
        institution = "Yeshiva University"

    field = params.get("fields")
    # If field is missing/empty, default to "computer science"
    if field is None or str(field).strip() == "":
        fields = "Computer Science"
    if isinstance(field, str):
        field = field

    year_min = params.get("year_min")
    year_max = params.get("year_max")

    return {
        "institution": institution,
        "fields":      fields,
        "year_min":   year_min,
        "year_max":   year_max,
    }


# Filter tool

def filter_tool(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:

    params = apply_defaults(params)
    result = df.copy()

    # Institution filter
    institution = params.get("institution")
    if institution and institution.lower() != "all":
        result = result[
            result["institution"].str.lower() == institution.lower()
        ]

    # Field filter
    field = params.get("field")
    if field and field.lower() != "all":
        result = result[result["field"].str.lower() == field.lower()]

    # Year range filters
    year_min = params.get("year_min")
    if year_min is not None and year_min != "":
        result = result[result["year"] >= int(year_min)]

    year_max = params.get("year_max")
    if year_max is not None and year_max != "":
        result = result[result["year"] <= int(year_max)]

    return result



# Analysis tool

def analyze_tool(df: pd.DataFrame, group_by: Union[str, List[str]]) -> pd.DataFrame:

    if isinstance(group_by, str):
        group_by = [group_by]

    out = (df.groupby(group_by)["paperid"].nunique().reset_index(name="num_papers").sort_values(group_by))
    return out



# Visualization tool

def bar_chart(df, x, y, title):
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": title,
        "data": {"values": df.to_dict(orient="records")},
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": x, "type": "ordinal", "sort": "ascending"},
            "y": {"field": y, "type": "quantitative"},
            "tooltip": [{"field": x}, {"field": y}]
        },
        "title": title
    }


def pie_chart(df, x, y, title):
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": title,
        "data": {"values": df.to_dict(orient="records")},
        "mark": {"type": "arc", "tooltip": True},
        "encoding": {
            "theta": {"field": y, "type": "quantitative"},
            "color": {"field": x, "type": "nominal"},
            "tooltip": [{"field": x}, {"field": y}]
        },
        "title": title
    }



################### Building LLM Agent ###########################

GEMINI_API_KEY = "...please into valid gemini api..."

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.0,
    max_retries=0,
)

resp = llm.invoke("Hello!")
print(resp.content)


# Building a planner LLM agent

PLANNER_SYSTEM_PROMPT = """
You are a planning agent for a SciSciNet data analysis chatbot.

The user will ask questions like:
- "show me the number of papers by year"
- "show me the number of papers by field between 2010 and 2020"
- "for Yeshiva University show me papers by year"

Your job is to output a valid JSON object with no extra text.

JSON schema:
{
  "group_by": "year" | "field" | ["year","field"] | ["field","year"],
  "chart_type": "bar" | "pie",
  "filters": {
    "institution": string or "Yeshiva University",
    "field": [string] or "computer science",
    "year_min": int or "2010",
    "year_max": int or "2024",
  }
}

Rules:
1. Choose group_by based on the user's request. If they mention "by year", use "year". If "by field" or a list of fields, use "field".
2. If the user mentions an institution, use that. Otherwise default to: "Yeshiva University".
3. If the user mentions a specific field or discipline (e.g., "biology", "AI", "economics"), set field to that exact string.
   If they do not mention a field, default to: "computer science".
4. If the user mentions a year range (e.g., 1990-2010), parse year_min and year_max as integers.
   If the user mentions only one year (e.g., "in 2015"), set both year_min and year_max to that year.
   If nothing about years is mentioned, set both year_min to 2010, year_max to 2024.
5. If the user says "bar chart", choose chart_type = "bar". If the user says "pie chart", choose chart_type = "pie".
   If the user does not specify, default chart_type = "bar".
6. Do not explain. Output only JSON file.
"""

def planner_for_user_query(user_query: str) -> Dict[str, Any]:
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]
    
    resp = llm.invoke(messages)
    print("RAW LLM RESPONSE:", repr(resp.content)) 
    
    raw = resp.content
    if raw is None:
        raise ValueError("LLM returned no content")

    raw = raw.strip()

    start = raw.find("{")
    end   = raw.rfind("}")
    
    json_str = raw[start:end+1]

    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("JSON FAILED TO PARSE:", json_str)
        raise e

    return plan




# Building a visualization LLM agent

VISUALIZER_SYSTEM_PROMPT = """
You are a visualization agent that outputs Vega-Lite v6 specifications.

You will receive JSON with the following structure:
{
  "data": [ { ... } ],
  "x_field": string,
  "y_field": string,
  "chart_type": "bar" | "pie",
  "title": string
}

Your job is to return a valid Vega-Lite v6 JSON specification.

Rules:
1. If chart_type="bar":
   - Use a bar chart with:
     {
       "mark": { "type": "bar", "tooltip": true },
       "encoding": {
         "x": { "field": x_field, "type": "ordinal" or "quantitative" },
         "y": { "field": y_field, "type": "quantitative" },
         "tooltip": [...]
       }
     }

2. If chart_type="pie":
   - Use an arc (pie) chart with:
     {
       "mark": { "type": "arc", "tooltip": true },
       "encoding": {
         "theta": { "field": y_field, "type": "quantitative" },
         "color": { "field": x_field, "type": "nominal" },
         "tooltip": [...]
       }
     }

3. Always include:
   "$schema": "https://vega.github.io/schema/vega-lite/v6.json"
   "data": { "values": [...] }
   "title": title

4. Respond with EXACTLY one JSON object, no markdown fences, no explanation.
"""


def planner_for_visualization(df_grouped: pd.DataFrame, x_field: str, y_field: str, chart_type: str, title: str) -> Dict[str, Any]:

    records = df_grouped[[x_field, y_field]].to_dict(orient="records")

    load = {
        "data": records,
        "x_field": x_field,
        "y_field": y_field,
        "chart_type": chart_type,
        "title": title,
    }

    messages = [
        SystemMessage(content=VISUALIZER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(load)),
    ]

    resp = llm.invoke(messages)
    raw = resp.content

    raw = str(raw).strip()
    print("RAW VISUALIZER OUTPUT:", repr(raw))

    start = raw.find("{")
    end   = raw.rfind("}")
    raw = raw[start:end+1]

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Failed to parse visualizer output:", e)
        print("Raw string was:\n", raw)
        raise

    return spec


######################### Chart Show ################################

def show_vega_spec(spec: dict):
    chart = alt.Chart.from_dict(spec)
    return chart




######################### Backend Pipeline ################################

def run_chat_turn(user_query: str, df_base: pd.DataFrame = Paper_field_inst):
    print(f"User: {user_query}\n")

    # 1. Planner: user → plan
    plan = planner_for_user_query(user_query)
    print("Plan:", json.dumps(plan, indent=2))

    group_by = plan["group_by"]
    filters = plan["filters"]

    # 2. Filter
    filtered = filter_tool(df_base, filters)
    grouped = analyze_tool(filtered, group_by)

    # 3. Build title
    field_text = filters.get("field") or "all fields"
    ym = filters.get("year_min") or ""
    yx = filters.get("year_max") or ""
    title = f"Number of papers by {group_by} – Yeshiva University"
    if ym or yx:
        title += f" ({ym}–{yx}), field: {field_text}"

    # 4. Get Vega-Lite spec from visualization LLM
    spec = planner_for_visualization(
        df_grouped=grouped,
        x_field=group_by,
        y_field="num_papers",
        chart_type=plan["chart_type"],
        title=title,
    )

    # 5. Display the chart
    chart = alt.Chart.from_dict(spec)

    display(grouped.head())
    display(chart)

    # 6. Return text
    num_papers = (
        len(filtered["paper_id"].unique())
        if "paper_id" in filtered.columns
        else len(filtered)
    )
    reply = f"Among {num_papers} papers, the results are grouped by {group_by} with filters {filters}." 

    # Return JSON-serializable content
    return {
        "reply": reply,
        "plan": plan,
        "title": title,
        "vega_spec": spec,  # Vega-Lite spec (dict)
        # optional preview table
        "grouped_preview": grouped.head(20).to_dict(orient="records"),
    }