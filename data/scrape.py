import praw
import pandas as pd
import os

reddit = praw.Reddit(client_id='AA7pptPLpjM-XA',
                     client_secret='YB6hIjE9no2YZ1h6eTyJqDCvvjPhAg',
                     user_agent='DaebakNLP')

subreddits_to_get = ['kpop']

PATH = 'reddit_scraped.csv'
COUNT = 500

for subs in subreddits_to_get:
    posts_dict = {
        "body": [],
        "class": [],
        "date": [],
        "score": [],
        "subreddit": [],
        "id": []
    }
    post_index = 0
    comments_index = 0

    for submission in reddit.subreddit(subs).hot(limit=COUNT):
        posts_dict['date'].append(submission.created_utc)
        posts_dict['score'].append(submission.score)
        posts_dict['subreddit'].append(subs)

        if submission.selftext == '':
            posts_dict['body'].append(submission.title)
        else:
            posts_dict['body'].append(submission.selftext)

        posts_dict['id'].append(submission.id)
        posts_dict['class'].append(0)

        post_index += 1
        if post_index >= COUNT:
            break
            
    pd.DataFrame(posts_dict).to_csv(PATH, index=False)

    print(f'Completed crawling {subs}, {post_index}/[post_count')

