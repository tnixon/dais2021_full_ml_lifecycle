# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Registry Webhooks
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step3.png?raw=true">
# MAGIC 
# MAGIC ### Supported Events
# MAGIC * Registered model created
# MAGIC * Model version created
# MAGIC * Transition request created
# MAGIC * Model version transitioned stage
# MAGIC 
# MAGIC ### Types of webhooks
# MAGIC * HTTP webhook -- send triggers to endpoints of your choosing such as slack, AWS Lambda, Azure Functions, or GCP Cloud Functions
# MAGIC * Job webhook -- trigger a job within the Databricks workspace
# MAGIC 
# MAGIC ### Use Cases
# MAGIC * Automation - automated introducing a new model to accept shadow traffic, handle deployments and lifecycle when a model is registered, etc..
# MAGIC * Model Artifact Backups - sync artifacts to a destination such as S3 or ADLS
# MAGIC * Automated Pre-checks - perform model tests when a model is registered to reduce long term technical debt
# MAGIC * SLA Tracking - Automatically measure the time from development to production including all the changes inbetween

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Webhooks
# MAGIC 
# MAGIC ___
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/webhooks2.png?raw=true" width = 600>

# COMMAND ----------

import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()

#dbutils.widgets.text("model_name", "")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transition Request Created
# MAGIC 
# MAGIC These fire whenever a transition request is created for a model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Trigger Job

# COMMAND ----------

# Which model in the registry will we create a webhook for?
model_name = dbutils.widgets.get("model_name")

trigger_job = json.dumps({
  "model_name": model_name,
  "events": ["TRANSITION_REQUEST_CREATED"],
  "description": "Trigger the ops_validation job when a model is moved to staging.",
  "status": "ACTIVE",
  "job_spec": {
    "job_id": "11507",    # This is our 05_ops_validation notebook
    "workspace_url": host,
    "access_token": token
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_job)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Notifications
# MAGIC 
# MAGIC Webhooks can be used to send emails, Slack messages, and more.  In this case we use Slack.  We also use `dbutils.secrets` to not expose any tokens, but the URL looks more or less like this:
# MAGIC 
# MAGIC `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX`
# MAGIC 
# MAGIC You can read more about Slack webhooks [here](https://api.slack.com/messaging/webhooks#create_a_webhook).

# COMMAND ----------

import urllib 
import json 

slack_webhook = dbutils.secrets.get("rk_webhooks", "slack")

# consider REGISTERED_MODEL_CREATED to run tests and autoamtic deployments to stages 
trigger_slack = json.dumps({
  "model_name": model_name,
  "events": ["TRANSITION_REQUEST_CREATED"],
  "description": "Notify the MLOps team that a model has moved from None to Staging.",
  "status": "ACTIVE",
  "http_url_spec": {
    "url": slack_webhook
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Version Transitioned Stage
# MAGIC 
# MAGIC These fire whenever a model successfully transitions to a particular stage.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Trigger Job

# COMMAND ----------

# Which model in the registry will we create a webhook for?
trigger_job = json.dumps({
  "model_name": model_name,
  "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
  "description": "Trigger the ops_validation job when a model is moved to staging.",
  "job_spec": {
    "job_id": "11507",
    "workspace_url": host,
    "access_token": token
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_job)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Notifications

# COMMAND ----------

import urllib 
import json 

trigger_slack = json.dumps({
  "model_name": model_name,
  "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
  "description": "Notify the MLOps team that a model has moved from None to Staging.",
  "http_url_spec": {
    "url": slack_webhook
  }
})

mlflow_call_endpoint("registry-webhooks/create", method = "POST", body = trigger_slack)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Manage Webhooks

# COMMAND ----------

# MAGIC %md
# MAGIC ##### List 

# COMMAND ----------

list_model_webhooks = json.dumps({"model_name": model_name})

mlflow_call_endpoint("registry-webhooks/list", method = "GET", body = list_model_webhooks)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Delete

# COMMAND ----------

# Remove a webhook
mlflow_call_endpoint("registry-webhooks/delete",
                     method="DELETE",
                     body = json.dumps({'id': 'c19ce9793122482d972862e4934ca416'}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find out more information on MLflow Model Registry?  
# MAGIC **A:** Check out <a href="https://mlflow.org/docs/latest/registry.html#concepts" target="_blank"> for the latest API docs available for Model Registry</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
