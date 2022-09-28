# Transaction class

class Transaction:

    number = 0
    address = ""
    phone = 0
    state = 'begin'
    name = ""

    '''
    Save order information
    '''

    def save(self):

        order = "Name: " + self.name + "Number: " + str(self.number) + ", Phone: " + str(self.phone) + ", Address: " + self.address
        f = open('data/order/orders.txt', 'a', errors='ignore', encoding='utf-8')
        f.writelines(str(order) + "\n")
        f.close()
