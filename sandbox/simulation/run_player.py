RL_TRAINING = False
RL_NAME = None

from datetime import datetime
import asyncio
import argparse
import pandas as pd
import traceback

from dragg_comp.player import PlayerHome
from dragg_comp.player_plot import PlayerPlotter
import time

REDIS_URL = "redis://localhost"

if __name__=="__main__":

	try:
		from submission.submission import *
		parser = argparse.ArgumentParser()

		#-db DATABSE -u USERNAME -p PASSWORD -size 20
		parser.add_argument("-r", "--redis", help="Redis host URL", default=REDIS_URL)

		args = parser.parse_args()

		env = PlayerHome(redis_url=args.redis)

		obs = env.reset()
		tic = datetime.now()
		print("Testing your agent...")
		for _ in range(env.num_timesteps):
			action = predict(env)
			env.step(action)

		asyncio.run(env.post_status("done"))
		print(env.score())
		toc = datetime.now()
		print(toc-tic)

		# Allow time for redis processes to stop
		time.sleep(5)
		
		p = PlayerPlotter(res_file='./outputs/', conf_file='./outputs/all_homes-10-config.json')
		p.main()

	except Exception as e:
		print('Exception raised by this code: {}'.format(e))
		print(traceback.format_exc())
		df = pd.DataFrame(
			{
				'l2_norm': ['Invalid Submission'],
				'contribution2peak': ['Invalid Submission'],
			}
		)
		df.to_csv('./outputs/score.csv')
