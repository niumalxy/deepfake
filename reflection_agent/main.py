import sys
import os
import json
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mq.consume import RabbitMQConsumer
from logger import logs
from reflection_agent.reflection import reflect_prompt

def create_reflection_callback(queue_name):
    def callback(ch, method, properties, body):
        message = json.loads(body)
        logs.info(f'[{queue_name}] Received message: {message}')
        try:
            task_id = message.get("task_id")
            if task_id:
                dump_type = "segment" if queue_name == "segment_queue" else "report"
                reflect_prompt(task_id, dump_type)
                logs.info(f'[{queue_name}] Message processed successfully: {message}')
            else:
                logs.error(f'[{queue_name}] No task_id found in message.')
            
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            logs.error(f'[{queue_name}] Message processing failed: {e}')
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
    return callback

def start_segment_consumer():
    consumer = RabbitMQConsumer('segment_queue')
    consumer.start_consuming(callback=create_reflection_callback('segment_queue'))

def start_report_consumer():
    consumer = RabbitMQConsumer('report_queue')
    consumer.start_consuming(callback=create_reflection_callback('report_queue'))

def main():
    logs.info("Starting Reflection Agent Consumers for both queues...")
    
    segment_thread = threading.Thread(target=start_segment_consumer, name="SegmentConsumerThread", daemon=True)
    report_thread = threading.Thread(target=start_report_consumer, name="ReportConsumerThread", daemon=True)
    
    segment_thread.start()
    report_thread.start()
    
    try:
        # Keep the main process alive
        segment_thread.join()
        report_thread.join()
    except KeyboardInterrupt:
        logs.info("Stopping Reflection Agent Main Process")
