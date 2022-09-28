# Main chatbot

from tkinter import *
from glove import glove
from sequence import sequence
from gru import gru
from transaction import Transaction
import re
from cosine_similarity import cosine_similarity

s = sequence()
t = Transaction()
c = cosine_similarity()
g = gru()
#w = glove() # This mehtod is too slow, strongly not recommended

method = s # Change 's', 'c', 'g', to try different intent matching methods

mode = 's'

# function of 'send' button
def send():

    global mode   

    # get user input
    user_response = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    # check to quit  
    if user_response == 'bye':
        exit()

    if user_response != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + user_response + '\n\n')
        ChatBox.config(foreground="black", font=("Times", 14 )) 

        # The talking mode, get a response
        if mode == 's':
            response = method.talk_response(user_response)
            if response == 'Navigating to transaction mode':
                ChatBox.insert(END, "Bot: " + response + '\n\n')
                mode = 't'

        # The transaction mode
        if mode == 't':
            # begin transaction
            if t.state == 'begin':
                response = "How many roses do you want to buy? (Please enter a number)"
                t.state = 'number'

            # get quantity
            elif t.state == 'number':
                number = re.findall(r'\b\d+\b', user_response)
                if number != []:
                    response = "You ordered %d" % int(number[0]) + "\n\nBot: Thanks! May I have your address please. "
                    t.state = 'address'
                    t.number = int(number[0])

                else:
                    response = "Invalid input, please enter a integer number"

            # get address
            elif t.state == 'address' :
                response = "Thanks! May I have your name please. "
                t.state = 'name'
                t.address = user_response

            # get name
            elif t.state == 'name' :
                response = "Thanks! May I have your phone number please. "
                t.state = 'phone'
                t.name = user_response

            # get phone number
            elif t.state == 'phone':
                phone = re.findall(r'\d+', user_response)
                if phone != []:
                    t.phone = int(phone[0])
                    response = "Order confirmed. Order information\nName: " + t.name + ", Number: " + str(t.number) + ", Phone: " + str(t.phone) + ", Address: " + t.address + '\n\n'
                    response = response + "We are selling red roses and I can answer your questions realted to our products!" + '\n\n'
                    response = response + "Hint: Try to say something like \"Make orders\" or \"How is the quality\"" + '\n\n'
                    response = response + "If you want to exit, please type \"bye\"" + '\n\n'
                    t.state = 'begin'
                    mode = 's'  
                    t.save()
                else:
                    response = "Invalid input, please enter again"
            
        ChatBox.insert(END, "Bot: " + response + '\n\n')           
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


if __name__ == '__main__':

    root = Tk()
    root.title("Chatbot")
    root.geometry("800x600")
    root.resizable(width=FALSE, height=FALSE)

    # Create Chat window
    ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font=("Times", 14))
    ChatBox.insert(END, "Bot: Hi, welcome to Rose Shop! We are selling red roses and I can answer your questions realted to our products!" + '\n\n')
    ChatBox.insert(END, "Hint: Try to say something like \"Make orders\" or \"How is the quality\"" + '\n\n')
    ChatBox.insert(END, "If you want to exit, please type \"bye\"" + '\n\n')
    ChatBox.config(state=DISABLED)
    
    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(root, command=ChatBox.yview)
    ChatBox['yscrollcommand'] = scrollbar.set

    # Create Button to send message
    SendButton = Button(root, font=("Times", 14, 'bold'), text="Send", width="10", height=10,
                            bd=0, bg="orange", activebackground="blue", fg='black',
                            command=send)

    # Create the box to enter message
    EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Times")

    # Place all components on the screen
    scrollbar.place(x=776, y=6, height=486)
    ChatBox.place(x=6, y=6, height=486, width=770)
    EntryBox.place(x=128, y=501, height=90, width=665)
    SendButton.place(x=6, y=501, height=90)

    root.mainloop()       

    

