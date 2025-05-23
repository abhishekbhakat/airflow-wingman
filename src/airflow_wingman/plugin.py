"""Plugin definition for Airflow Wingman."""

from airflow.plugins_manager import AirflowPlugin
from flask import Blueprint

from airflow_wingman.views import WingmanView

bp = Blueprint(
    "wingman",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/wingman",
)

v_appbuilder_view = WingmanView()
v_appbuilder_package = {
    "name": "Wingman",
    "category": "AI",
    "view": v_appbuilder_view,
}


class WingmanPlugin(AirflowPlugin):
    """Airflow plugin for Wingman chat interface."""

    name = "wingman"
    flask_blueprints = [bp]
    appbuilder_views = [v_appbuilder_package]
