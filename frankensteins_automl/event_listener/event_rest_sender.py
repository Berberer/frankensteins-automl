import requests
import uuid
from pubsub import pub
from frankensteins_automl.event_listener import event_topics

relevant_topics = [event_topics.MCTS_TOPIC, event_topics.SEARCH_GRAPH_TOPIC]
url = None


def activate(rest_url):
    global url
    url = rest_url
    for topic in relevant_topics:
        pub.subscribe(_listener, topic)


def _listener(payload, topic=pub.AUTO_TOPIC):
    payload["message_id"] = str(uuid.uuid1())
    requests.post(url, json=payload)
