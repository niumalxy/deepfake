import yaml
with open('mq/conf/conf.yml', 'r') as f:
    conf = yaml.safe_load(f)
    rabbitmq = conf['rabbitmq']
    host = str(rabbitmq['host'])
    port = int(rabbitmq['port'])
    username = str(rabbitmq['username'])
    password = str(rabbitmq['password'])
