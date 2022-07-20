from matplotlib import pyplot as plt



class Visualizer:
    def vectors_2D(vectors):
        plt_figure = plt.figure()
        x = [vector[0] for vector in vectors]
        y = [vector[1] for vector in vectors]
        plt.plot(x, y, 'o')
        plt_figure.show()
        plt.xlim(0, max(x)+1)
        plt.ylim(0, max(y)+1)
        plt.show()

    def vectors_2D_multiple_classes(vector_dict):
        plt_figure = plt.figure()
        for cluster_vectors in vector_dict.values():
            x = [cluster_vector[0] for cluster_vector in cluster_vectors]
            y = [cluster_vector[1] for cluster_vector in cluster_vectors]
            plt.plot(x, y, 'o')
        plt_figure.show()
        plt.show()