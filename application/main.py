# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import url_for
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from google.appengine.api import app_identity


credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials)
project = app_identity.get_application_id()
model_name = os.getenv('MODEL_NAME', 'candy')
version_name = os.getenv('VERSION_NAME', 'ml_on_gcp')


app = Flask(__name__)


def get_prediction(features):
  input_data = {'instances': [features]}
  parent = 'projects/%s/models/%s/versions/%s' % (project, model_name, version_name)
  prediction = api.projects().predict(body=input_data, name=parent).execute()
  return prediction['predictions'][0]['predictions'][0]


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/form')
def input_form():
  return render_template('form.html')


@app.route('/api/predict', methods=['POST'])
def predict():

  def categorical2int(val):
      options = {'true': 1, 'false': 0}
      return options[val]
      
  data = json.loads(request.data.decode())
  mandatory_items = ['chocolate', 'fruity', 'peanutyalmondy', 'crispedricewafer',
                      'hard', 'bar', 'pricepercent']
  # for item in mandatory_items:
  #   if item not in data.keys():
  #     return jsonify({'result': 'Set all items.'})

  features = {}
  features['hashname'] = 'nokey'
  features['chocolate'] = categorical2int(data['chocolate'])
  features['fruity'] = categorical2int(data['fruity'])
  features['peanutyalmondy'] = categorical2int(data['peanutyalmondy'])
  features['crispedricewafer'] = categorical2int(data['crispedricewafer'])
  features['hard'] = categorical2int(data['hard'])
  features['bar'] = categorical2int(data['bar'])
  features['pricepercent'] = float(data['pricepercent'])

  prediction = get_prediction(features)
  return jsonify({'result': '{:.2f} lbs.'.format(prediction)})
