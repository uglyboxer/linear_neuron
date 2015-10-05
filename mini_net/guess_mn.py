import mnist
from network import Network

def main():
    dataset = 'training'
    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    training_set, answers = mnist.read(target_values)
    dataset = 'testing'
    testing_set, answers_to_test = mnist.read(target_values, 'testing')

    epoch = 100

    # For all inputs
    network = Network(target_values, training_set, answers, epoch, testing_set,
                      answers_to_test, validation_set, validation_answers,
                      images)
    network.learn_run()
    network.report_results(network.run_unseen())
    # network.report_results(network.run_unseen(True), True)

if __name__ == '__main__':
    main()