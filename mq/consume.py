import pika
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mq.conf.conf import rabbitmq, host, port, username, password


class RabbitMQConsumer:
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
        self.channel.basic_qos(prefetch_count=1)

    def callback(self, ch, method, properties, body):
        message = json.loads(body)
        print(f'收到消息: {message}')
        try:
            self.process_message(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f'消息处理完成: {message}')
        except Exception as e:
            print(f'消息处理失败: {e}')
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def process_message(self, message):
        task_type = message.get('type')
        task_id = message.get('task_id')
        data = message.get('data')
        
        print(f'处理任务类型: {task_type}')
        print(f'任务ID: {task_id}')
        print(f'数据: {data}')
        
        if task_type == 'image_segment':
            self._handle_image_segment(data)
        elif task_type == 'content_analysis':
            self._handle_content_analysis(data)
        elif task_type == 'report_generation':
            self._handle_report_generation(data)
        else:
            print(f'未知任务类型: {task_type}')

    def _handle_image_segment(self, data):
        print(f'开始处理图像分割: {data}')

    def _handle_content_analysis(self, data):
        print(f'开始处理内容分析: {data}')

    def _handle_report_generation(self, data):
        print(f'开始生成报告: {data}')

    def start_consuming(self, callback=None):
        if callback:
            self.callback = callback
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.callback
        )
        print('等待消息... 按 Ctrl+C 退出')
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print('退出消费')
            self.close()

    def close(self):
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            print('RabbitMQ 连接已关闭')


def default_callback(ch, method, properties, body):
    consumer = RabbitMQConsumer(queue_name='test_queue')
    consumer.callback(ch, method, properties, body)


def main():
    consumer = RabbitMQConsumer(queue_name='test_queue')
    consumer.start_consuming()


if __name__ == '__main__':
    main()
