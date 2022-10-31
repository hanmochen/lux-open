class Worker(object):
    def __init__(self):
        pass
    
    def move(self, id, dir):
        """
        return the command to move unit in the given direction
        """
        return "m {} {}".format(id, dir)

    def transfer(self, id, dest_id, resourceType, amount):
        """
        return the command to transfer a resource from a source unit to a destination unit as specified by their ids
        """
        return "t {} {} {} {}".format(id, dest_id, resourceType, amount)

    def build_city(self, id):
        """
        return the command to build a city right under the worker
        """
        return "bcity {}".format(id)

    def pillage(self, id):
        """
        return the command to pillage whatever is underneath the worker
        """
        return "p {}".format(id)

class Cart(object):
    def __init__(self):
        pass

    def move(self, id, dir):
        """
        return the command to move unit in the given direction
        """
        return "m {} {}".format(id, dir)

    def transfer(self, id, dest_id, resourceType, amount):
        """
        return the command to transfer a resource from a source unit to a destination unit as specified by their ids
        """
        return "t {} {} {} {}".format(id, dest_id, resourceType, amount)

class Citytile(object):
    def __init__(self):
        pass

    def research(self, pos_x, pos_y):
        """
        returns command to ask this tile to research this turn
        """
        return "r {} {}".format(pos_x, pos_y)
    def build_worker(self, pos_x, pos_y):
        """
        returns command to ask this tile to build a worker this turn
        """
        return "bw {} {}".format(pos_x, pos_y)
    def build_cart(self, pos_x, pos_y):
        """
        returns command to ask this tile to build a cart this turn
        """
        return "bc {} {}".format(pos_x, pos_y)
