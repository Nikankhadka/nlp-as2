"""Conversation memory — tracks last 3 turns + last topic for follow-up questions.
Why: Lets the bot answer "What about the service?" after someone asked about food,
without the user needing to repeat context."""

from collections import deque


class ConversationMemory:
    def __init__(self, max_turns=3):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
        self.last_topic = None

    def add_exchange(self, user_msg, bot_msg, intent, topic=None):
        self.history.append({
            'user': user_msg,
            'bot': bot_msg,
            'intent': intent,
            'topic': topic
        })
        if topic:
            self.last_topic = topic

    def get_context(self):
        if not self.history:
            return ''
        recent = self.history[-1]
        return (f'User just asked: "{recent["user"]}". '
                f'The intent was {recent["intent"]}. '
                f'I responded about: {recent["topic"] or "N/A"}.')

    def get_previous_intent(self):
        if self.history:
            return self.history[-1]['intent']
        return None

    def get_previous_topic(self):
        return self.last_topic
