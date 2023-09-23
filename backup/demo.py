# just run once. this will load the model. super slow and take GPU memory.
from gpt_neo_completion import gpt_neo_completion
import time
import torch
print(torch.cuda.is_available())

table_prompt = """
CREATE TABLE hotel(
name text,
pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
type text CHECK (type IN (hotel, guest house)),
parking text CHECK (parking IN (dontcare, yes, no)),
book_stay int,
book_day text,
book_people int,
area text CHECK (area IN (dontcare, centre, east, north, south, west)),
stars int CHECK (stars IN (dontcare, 0, 1, 2, 3, 4, 5)),
internet text CHECK (internet IN (dontcare, yes, no))
)
/*
4 example rows:
SELECT * FROM hotel LIMIT 4;
name pricerange type parking book_stay book_day book_people area stars internet
a and b guest house moderate guest house dontcare 3 friday 5 east 4 yes
ashley hotel expensive hotel yes 2 thursday 5 north 5 yes
el shaddia guest house cheap guest house yes 5 friday 2 centre dontcare no
express by holiday inn cambridge dontcare guest house yes 3 monday 2 east dontcare no
*/

CREATE TABLE train(
destination text,
departure text,
day text,
book_people int,
leaveat text,
arriveby text
)
/*
3 example rows:
SELECT * FROM train LIMIT 3;
destination departure day book_people leaveat arriveby
london kings cross cambridge monday 6 dontcare 05:51
cambridge stansted airport dontcare 1 20:24 20:52
peterborough cambridge saturday 2 12:06 12:56
*/

CREATE TABLE attraction(
name text,
area text CHECK (area IN (dontcare, centre, east, north, south, west)),
type text,
)
/*
4 example rows:
SELECT * FROM attraction LIMIT 4;
name area type
abbey pool and astroturf pitch centre swimming pool
adc theatre centre theatre
all saints church dontcare architecture
castle galleries centre museum
*/

CREATE TABLE restaurant(
name text,
food text,
pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
area text CHECK (area IN (centre, east, north, south, west)),
book_time text,
book_day text,
book_people int
)
/*
5 example rows:
SELECT * FROM restaurant LIMIT 5;
name food pricerange area book_time book_day book_people
pizza hut city centre italian dontcare centre 13:30 wednesday 7
the missing sock international moderate east dontcare dontcare 2
golden wok chinese moderate north 17:11 friday 4
cambridge chop house dontcare expensive center 08:43 monday 5
darrys cookhouse and wine shop modern european expensive center 11:20 saturday 8
*/

CREATE TABLE taxi(
destination text,
departure text,
leaveat text,
arriveby text
)
/*
3 example rows:
SELECT * FROM taxi LIMIT 3;
destination departure leaveat arriveby
copper kettle royal spice 14:45 15:30
magdalene college university arms hotel dontcare 15:45
lovell lodge da vinci pizzeria 11:45 dontcare
*/

-- Using valid SQLite, answer the following multi-turn conversational questions for the tables provided above.
"""


current_test_turn = """[context] attraction-area: east
[system] how about cambridge university ? it s a university on the east side of town , and they have no entrance fee .
Q: [user]okay, I also need a Korean restaurant in the same area. Can you recommend one?"""


examples = """
Example #1
[context] attraction-area: centre, attraction-type: museum, train-departure: cambridge, train-day: friday, train-
arrive_by_time: 12:45, train-book people: 6, train-destination: leicester
[system] i recommend castle galleries located at unit su43 , grande arcade , saint andrews street . their phone number is
01223307402 . is there anything else i can help you with ?
Q: [user] excellent , can you give me the postcode ?
SQL: SELECT * FROM attraction WHERE name = castle galleries;


Example #2
[context] attraction-type: museum, restaurant-book day: wednesday, restaurant-book people: 7, restaurant-name: loch fyne,
restaurant-book time: 16:30, attraction-area: west
[system] i would suggest cafe jello gallery located at cafe jello gallery , 13 magdalene street . they have free entry .
Q: [user] okay great ! what is their phone number please ?
SQL: SELECT * FROM attraction WHERE name = cafe jello gallery;


Example #3
[context] attraction-area: centre, attraction-type: museum
[system] the broughton house gallery is in the centre , and it has no entrance fee .
Q: [user] may i have the telephone number please ?
SQL: SELECT * FROM attraction WHERE name = broughton house gallery;


Example #4
[context] train-arrive_by_time: 21:30, train-destination: leicester, train-day: thursday, train-departure: cambridge
[system] how many tickets please ?
Q: [user] i do not need to make the reservation now . thank you though . i would like the address for cambridge contemporary
art please .
SQL: SELECT * FROM attraction WHERE name = cambridge contemporary art;


Example #5
[context] attraction-area: east
[system] i like the cambridge artworks it s a museum at 5 greens road and it has free admission .
Q: [user] that sounds , good , what is the postcode ?
SQL: SELECT * FROM attraction WHERE name = cambridge artworks;


"""


prompt = table_prompt + examples + "Example #6\n" + \
    current_test_turn + "\nSQL: SELECT * FROM"


print(gpt_neo_completion(prompt))
