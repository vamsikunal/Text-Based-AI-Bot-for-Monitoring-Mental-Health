def main():
    import SentimentAnalysis.analyzingData as td
    from collections import defaultdict

    with open('SpeechToText.txt') as f:
        lines=f.read().split('\n')

    user=[line.split(':')[0].strip() for line in lines]
    chat=[line.split(':')[1].strip() for line in lines]

    x=list(set(user))
    #print(x)
    unique_user_count=len(x)
    #print(unique_user_count)

    d = defaultdict(lambda: len(d))
    #print(x)
    #print(unique_user)

    list_user_sent=[[] for i in range(unique_user_count)]
    sentCount=[[] for i in range(unique_user_count)]
    sent=[[] for i in range(unique_user_count)]
    feedback = [[] for i in range(unique_user_count)]
    #feedbackCount = [[] for i in range(unique_user_count)]
    for i in range(len(chat)):
        f=open('read','w')
        f.write(chat[i])
        f.close()
        pred=td.test() #Actual Prediction Result as a List //Calling main SVM
        #print(pred)
        res = int("".join(map(str, pred[0])))
        if res==0:
            ans='Negative'
            #print(feedback)
        else:
            ans='Positive'
        list_user_sent[d[user[i]]].append(ans)
        feedback[d[user[i]]].append(pred[1])

    for i in range(len(x)):
        sentCount[d[x[i]]]=len(list_user_sent[d[x[i]]])
        #print('User',x[i],':',list_user_sent[d[x[i]]],'Sentiment Count',sentCount[d[x[i]]])

    for i in range(len(x)):
        k=sentCount[d[x[i]]]
        for l in range(k):
            sent[d[x[i]]].append(l)

    #print(feedback[d['A']])
    #print(feedback[d['B']])
    index=0
    for i in range(len(chat)):
        index=sent[d[user[i]]][0]
        sent[d[user[i]]].pop(0)
        return (user[i],chat[i],list_user_sent[d[user[i]]][index])
        if list_user_sent[d[user[i]]][index] == 'Negative':
           return (feedback[d[user[i]]][index])