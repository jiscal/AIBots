# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
from typing import Dict, Text, Any, List, Union

from rasa_sdk import Tracker, Action
from rasa_sdk.events import UserUtteranceReverted, Restarted, SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction

from rasa_sdk.forms import FormAction
import requests
from requests import (
    ConnectionError,
    HTTPError,
    TooManyRedirects,
    Timeout
)

KEY='2f0d85dae4dd40948b41a22866485bd5'
Code_API='https://geoapi.qweather.com/v2/city/lookup?'
API='https://devapi.qweather.com/v7/weather/3d?'

class WeatherForm(FormAction):

    def name(self) -> Text:
        """Unique identifier of the form"""

        return "weather_form"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:

        return ["date_time", "address"]

    def submit(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
    ) -> List[Dict]:
        address = tracker.get_slot('address')
        print(address)
        date_time = tracker.get_slot('date_time')
        print(date_time)
        date_time_number = text_date_to_number_date(date_time)
        print(date_time_number)
        if isinstance(date_time_number, str):  # parse date_time failed
            dispatcher.utter_message("暂不支持查询 {} 的天气".format([address, date_time_number]))
        if date_time_number>3:
            dispatcher.utter_message('超过三天为收费版，需要付费')
        else:
            try:
                print('start')
                #address='hangzhou'
                #print(address)
                city_code = requests.get(Code_API, params={
                    'key': KEY,
                    'location': address
                }, timeout=2
                )
                #print(city_code.json())
                code = city_code.json()['location'][0]['id']
                print(code)
                weather = requests.get(API, params={
                    'key': KEY,
                    'location': code
                }, timeout=2
                )
                #print(weather.json())
                result=weather.json()['daily'][date_time_number]
                print(result)
            except (ConnectionError, HTTPError, TooManyRedirects, Timeout) as e:
                text_message = "{}".format(e)
            else:
                text_message_tpl = """
                    {} {} ({}) 的天气情况为：白天：{}；夜晚：{}；温度：{}-{} °C
                """
                text_message = text_message_tpl.format(
                    address,
                    date_time,
                    result['fxDate'],
                    result['textDay'],
                    result['textNight'],
                    result['tempMax'],
                    result["tempMin"],
                )
                print(text_message)
                dispatcher.utter_message(text_message)
            return []


def text_date_to_number_date(text_date):
    if text_date == "今天":
        return 0
    if text_date == "明天":
        return 1
    if text_date == "后天":
        return 2

    # Not supported by weather API provider freely
    if text_date == "大后天":
        # return 3
        return text_date

    if text_date.startswith("星期"):
        # @todo: using calender to compute relative date
        return text_date

    if text_date.startswith("下星期"):
        # @todo: using calender to compute relative date
        return text_date

    # follow APIs are not supported by weather API provider freely
    if text_date == "昨天":
        return text_date
    if text_date == "前天":
        return text_date
    if text_date == "大前天":
        return text_date
