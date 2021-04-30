"""
e.g. run:
job_id=$(python redis-queue-test.py https://google.com/)
python redis-queue-test.py https://amazon.com/ $job_id
rq worker -u $REDIS_URL
"""

#%%
import os
import sys
from rq import Queue
from redis import Redis
from redis_queue_module import count_words_at_url

q = Queue(connection=Redis(
    host=os.environ.get('REDIS_HOST'),
    port=os.environ.get('REDIS_PORT'),
    password=os.environ.get('REDIS_PASSWORD'),
    ssl=True
))

dep = sys.argv[2] if len(sys.argv) > 2 else None
job = q.enqueue(
    count_words_at_url,
    sys.argv[1],
    depends_on=dep,
    result_ttl='1h')
print(job.id)
