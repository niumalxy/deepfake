from mq.produce import RabbitMQProducer
from entity.dump_type import DumpType

def reflection_produce(message, type: DumpType):
    queue_name = ""
    if type == DumpType.SEGMENT:
        queue_name = 'segment_queue'
    elif type == DumpType.REPORT:
        queue_name = 'report_queue'
    
    producer = RabbitMQProducer(queue_name=queue_name)
    producer.publish(message)
    producer.close()
