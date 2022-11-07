import multiprocessing


class Apple():
    def __init__(self, q):
        self.q = q

    def square_list(self, mylist):
        for num in mylist:
            self.q.put(num * num)


def print_list(q):
    while not q.empty():
        print(q.get())


if __name__ == '__main__':
    q = multiprocessing.Queue()

    apple = Apple(q)

    p1 = multiprocessing.Process(target=apple.square_list, args=([1, 2, 3, 4],))
    p2 = multiprocessing.Process(target=print_list, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
