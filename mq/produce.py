import pika
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mq.conf.conf import rabbitmq, host, port, username, password


class RabbitMQProducer:
    def __init__(self, queue_name):
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        credentials = pika.PlainCredentials(username, password)
        parameters = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)

    def publish(self, message):
        try:
            if self.connection.is_closed:
                self._connect()
            self.channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type='application/json'
                )
            )
            print(f'消息已发送: {message}')
            return True
        except Exception as e:
            print(f'消息发送失败: {e}')
            return False

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            print('RabbitMQ 连接已关闭')


def main():
    producer = RabbitMQProducer(queue_name='test_queue')
    
    test_messages = [
        {'task_id': '1', 'type': 'image_segment', 'data': 'test_image_1.jpg'},
        {'task_id': '2', 'type': 'content_analysis', 'data': 'test_image_2.jpg'},
        {'task_id': '3', 'type': 'report_generation', 'data': 'test_result'}
    ]
    
    for msg in test_messages:
        producer.publish(msg)
    
    producer.close()


if __name__ == '__main__':
    main()
