from trello import TrelloClient
import datetime
client = TrelloClient(
    api_key='44010a512ef70ddfa9f0a319df5ebc4c',
    api_secret='4e311259dd3384ed3af45364d4f9921bd94fe71d9a339d1d76432a0acb411ebc',
    token='fe0cfa5473573f8762afc931c3054ccf7a7f0a9725fc81b1608ee495d91d6983'
    #token_secret='87f402413b31d55ddf22d610f96ead760944c39632f47cc77e5b2a08d811255a'
)

all_boards = client.list_boards()
last_board = all_boards[-1]
#print(last_board.list_lists())

games_board = last_board.list_lists()[0].list_cards()[0]

#print(games_board.checklists[0].items)
#for k in last_board.list_lists():
#    print(k.list_cards().)
print(f"Checklist title is {games_board.checklists[0].name}")

completed = []
notCompleted = []

for k in games_board.checklists[0].items:
    if k['state'] == 'complete':
        completed.append(k['name'])
    else:
        notCompleted.append(k['name'])
        
print(f"\nCompleted Tasks are {completed}") 

   
print(f"\nNot Completed Tasks are {notCompleted}")

print(f"\nCompletion in percents {len(completed) / len(games_board.checklists[0].items) * 100:.0f}")