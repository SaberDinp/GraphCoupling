from copy import deepcopy

import numpy as np
class WarmStart:
    def __init__(self):
        pass

    def get_rows_for_a_star(self, head, neighbors, n):
        row_1, row_2, row_3 = [], [], []
        for i in range(n):
            if i == head:
                row_1.append(-1)
                row_2.append(-1)
                row_3.append(1)
            elif i in neighbors:
                row_1.append(-1)
                row_2.append(1)
                row_3.append(-1)
            else:
                row_1.append(1)
                row_2.append(1)
                row_3.append(1)
        # returns the rows with corersponding weights. The last value is the weight of row of ones
        return row_1, row_2, row_3, 0.25, -0.25, -0.25, 0.25

    def get_rows_for_a_double_star(self, head1, neighbors1, head2, neighbors2, n):
        neighbors1 = set(neighbors1)
        neighbors2 = set(neighbors2)

        common_neighbors = neighbors1.intersection(neighbors2)
        exclusive_neighbors1 = neighbors1.difference(common_neighbors)
        exclusive_neighbors2 = neighbors2.difference(common_neighbors)

        row_1, row_2, row_3, row_4, row_5 = [], [], [], [], []
        for i in range(n):
            if i == head1:
                row_1.append(1)
                row_2.append(1)
                row_3.append(1)
                row_4.append(1)
                row_5.append(1)
            elif i == head2:
                row_1.append(1)
                row_2.append(1)
                row_3.append(1)
                row_4.append(-1)
                row_5.append(-1)
            elif i in exclusive_neighbors1:
                row_1.append(1)
                row_2.append(1)
                row_3.append(-1)
                row_4.append(1)
                row_5.append(-1)
            elif i in common_neighbors:
                row_1.append(1)
                row_2.append(-1)
                row_3.append(-1)
                row_4.append(-1)
                row_5.append(-1)
            elif i in exclusive_neighbors2:
                row_1.append(-1)
                row_2.append(-1)
                row_3.append(-1)
                row_4.append(-1)
                row_5.append(1)
            else:
                row_1.append(-1)
                row_2.append(1)
                row_3.append(-1)
                row_4.append(1)
                row_5.append(1)

        # returns the rows with corersponding weights. The last value is the weight of row of ones
        return row_1, row_2, row_3, row_4, row_5, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25

    def get_clique_rows(self, vertices, n):
        rows = []
        for i in range(len(vertices)):
            vertex = vertices[i]
            rows.append([-1 if i == vertex else 1 for i in range(n)])
        rows.append([-1 if i in vertices else 1 for i in range(n)])
        return rows, [-0.25 for _ in range(len(vertices))] + [0.25, (len(vertices) - 1) / 4]




    """
        Makes the first element of all rows of P to be 1. Then sorts the rows so that they are in the increasing order.
    """
    def convert_to_standard_form(self, p, w):
        new_p = deepcopy(p)
        for i in range(len(p)):
            if new_p[i][0] == -1:
                new_p[i] *= -1

        sort_indices = np.argsort([sum(2 ** j * (new_p[i][j] + 1) / 2 for j in range(len(p[0]))) for i in range(len(new_p))])

        new_p = new_p[sort_indices]
        w = np.array(w)[sort_indices]
        return new_p, w


    def remove_redundant_rows(self, p, w):
        n = len(p[0])
        new_p = np.zeros((0, n))
        new_ws = []
        row_nums = [sum(2 ** j * (p[r][j] + 1) / 2 for j in range(len(p[0]))) for r in range(len(p))]
        row_nums_set = set(row_nums)
        for row_num in sorted(row_nums_set):
            new_w = 0
            row = []
            for i in range(len(p)):
                if row_nums[i] == row_num:
                    new_w += w[i]
                    row = p[i]
            new_p = np.vstack((new_p, row))
            new_ws.append(new_w)
        return new_p, new_ws

    def remove_rows_with_weight_zero(self, p, w):
        n = len(p[0])
        new_p = np.zeros((0, n))
        new_ws = []
        for i in range(len(p)):
            row = p[i]
            weight = w[i]
            if weight != 0:
                new_p = np.vstack((new_p, row))
                new_ws.append(weight)

        return new_p, new_ws

    """
        Input: adjacency matrix A
        Output: a feasible P and W for it using the union of stars construction from Joel Rajakumar's paper: 
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.022606
    """
    def get_union_of_stars_feasible_solution(self, A, remove_redundant_rows, remove_rows_with_zero_weight, choose_stars_greedily):
        n = len(A[0])
        p = np.zeros((0, n))
        w = []
        w_row_of_ones = 0

        if choose_stars_greedily:
            remaining_vertices = list(range(n))
            while len(remaining_vertices) > 0:
                head = sorted(remaining_vertices, key=lambda v: -sum(A[v][u] for u in remaining_vertices))[0]
                neighbors = [v for v in remaining_vertices if A[head][v] == 1]
                if len(neighbors) != 0:
                    row_1, row_2, row_3, w_1, w_2, w_3, w_ones = self.get_rows_for_a_star(head, neighbors, n)
                    p = np.vstack((p, row_1, row_2, row_3))
                    w += [w_1, w_2, w_3]
                    w_row_of_ones += w_ones
                remaining_vertices.remove(head)
        else:
            for i in range(n):
                neighbors = [j for j in range(i + 1, n) if A[i][j] == 1]
                if len(neighbors) != 0:
                    row_1, row_2, row_3, w_1, w_2, w_3, w_ones = self.get_rows_for_a_star(i, neighbors, n)
                    p = np.vstack((p, row_1, row_2, row_3))
                    w += [w_1, w_2, w_3]
                    w_row_of_ones += w_ones

        p = np.vstack((p, [1 for j in range(n)]))
        w.append(w_row_of_ones)
        p, w = self.convert_to_standard_form(p, w)
        if remove_redundant_rows:
            p, w = self.remove_redundant_rows(p, w)
        if remove_rows_with_zero_weight:
            p, w = self.remove_rows_with_weight_zero(p, w)
        w = np.diag(w)
        return (p + 1) / 2, w
        # return p, w

    """"
        Input: adjacency matrix A
        Output: a feasible P and W for it using out novel theoretical upper bound: 2.5 + 2
    """
    def try_finding_double_star(self, A, remaining_vertices, p, w, w_row_of_ones, n, choose_double_stars_greedily):
        if choose_double_stars_greedily:
            best_i = None
            best_j = None
            best_double_star_value = -1
            for i in range(len(remaining_vertices)):
                for j in range(i + 1, len(remaining_vertices)):
                    head1 = remaining_vertices[i]
                    head2 = remaining_vertices[j]
                    if A[head1][head2] == 0:
                        neighbors1 = [u for u in remaining_vertices if A[head1][u] == 1]
                        neighbors2 = [u for u in remaining_vertices if A[head2][u] == 1]
                        if len(neighbors1) + len(neighbors2) > best_double_star_value:
                            best_i = i
                            best_j = j
            if best_i is not None:
                head1 = remaining_vertices[best_i]
                head2 = remaining_vertices[best_j]
                neighbors1 = [u for u in remaining_vertices if A[head1][u] == 1]
                neighbors2 = [u for u in remaining_vertices if A[head2][u] == 1]
                row_1, row_2, row_3, row_4, row_5, w_1, w_2, w_3, w_4, w_5, w_ones = self.get_rows_for_a_double_star(
                    head1, neighbors1, head2, neighbors2, n)
                if len(neighbors1) + len(neighbors2) == 0:
                    return p, w, w_row_of_ones, list(set(remaining_vertices).difference(set([head1, head2])))
                # elif len(neighbors1) == 0 and len(neighbors2) != 0:
                #     row_1, row_2, row_3, w_1, w_2, w_3, w_ones = self.get_rows_for_a_star(head2, neighbors2, n)
                #     return np.vstack((p, row_1, row_2, row_3)), w + [w_1, w_2, w_3], w_row_of_ones + w_ones, list(set(remaining_vertices).difference(set([head1, head2])))
                # elif len(neighbors1) != 0 and len(neighbors2) == 0:
                #     row_1, row_2, row_3, w_1, w_2, w_3, w_ones = self.get_rows_for_a_star(head1, neighbors1, n)
                #     return np.vstack((p, row_1, row_2, row_3)), w + [w_1, w_2, w_3], w_row_of_ones + w_ones, list(
                #         set(remaining_vertices).difference(set([head1, head2])))
                else:
                    return np.vstack((p, row_1, row_2, row_3, row_4, row_5)), w + [w_1, w_2, w_3, w_4,
                                                                                   w_5], w_row_of_ones + w_ones, list(
                        set(remaining_vertices).difference(set([head1, head2])))


        else:
            for i in range(len(remaining_vertices)):
                for j in range(i + 1, len(remaining_vertices)):
                    head1 = remaining_vertices[i]
                    head2 = remaining_vertices[j]
                    if A[head1][head2] == 0:
                        neighbors1 = [u for u in remaining_vertices if A[head1][u] == 1]
                        neighbors2 = [u for u in remaining_vertices if A[head2][u] == 1]
                        if len(neighbors1) + len(neighbors2) == 0:
                            return p, w, w_row_of_ones, list(set(remaining_vertices).difference(set([head1, head2])))
                        else:
                            row_1, row_2, row_3, row_4, row_5, w_1, w_2, w_3, w_4, w_5, w_ones = self.get_rows_for_a_double_star(
                                head1, neighbors1, head2, neighbors2, n)
                            return np.vstack((p, row_1, row_2, row_3, row_4, row_5)), w + [w_1, w_2, w_3, w_4, w_5], w_row_of_ones + w_ones, list(set(remaining_vertices).difference(set([head1, head2])))
    """greedy choice of double stars means that at each step we will choose the double star with maximum number of edges"""
    def get_union_of_double_stars_feasible_solution(self, A, remove_redundant_rows, remove_rows_with_zero_weight, choose_double_stars_greedily):
        n = len(A[0])
        p = np.zeros((0, n))
        w = []
        w_row_of_ones = 0

        remaining_vertices = list(range(n))
        while len(remaining_vertices) >= 2:
            ans = self.try_finding_double_star(A, remaining_vertices, p, w, w_row_of_ones, n, choose_double_stars_greedily)
            if ans is None:
                break
            else:
                p = ans[0]
                w = ans[1]
                w_row_of_ones = ans[2]
                remaining_vertices = ans[3]

        if len(remaining_vertices) >= 2:
            # construct clique
            clique_p_rows, clique_w_weights = self.get_clique_rows(remaining_vertices, n)
            p = np.vstack((p, clique_p_rows))
            p = np.vstack((p, [1 for j in range(n)]))
            w += clique_w_weights

        p = np.vstack((p, [1 for j in range(n)]))
        w.append(w_row_of_ones)

        p, w = self.convert_to_standard_form(p, w)
        if remove_redundant_rows:
            p, w = self.remove_redundant_rows(p, w)
        if remove_rows_with_zero_weight:
            p, w = self.remove_rows_with_weight_zero(p, w)
        w = np.diag(w)
        # return p, w
        return (p + 1) / 2, w



if __name__ == '__main__':
    # A = [
    #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]

    # A = [
    #     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]
    # A_12 = [
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # ]
    # A_25 = [
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # ]
    # A = [[0, 0, 0],
    #      [0, 0, 1],
    #      [0, 1, 0]]

    warm_start = WarmStart()
    single_non_greedy = []
    single_greedy = []
    double_non_greedy = []
    double_greedy = []

    for n in range(1, 50):
        A = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1  # Ensure symmetry
        p, w = warm_start.get_union_of_stars_feasible_solution(A, remove_redundant_rows=True, remove_rows_with_zero_weight=True, choose_stars_greedily=False)
        single_non_greedy.append(len(p))
        p, w = warm_start.get_union_of_stars_feasible_solution(A, remove_redundant_rows=True, remove_rows_with_zero_weight=True, choose_stars_greedily=True)
        single_greedy.append(len(p))
        p, w = warm_start.get_union_of_double_stars_feasible_solution(A, remove_redundant_rows=True, remove_rows_with_zero_weight=True, choose_double_stars_greedily=False)
        double_non_greedy.append(len(p))
        p, w = warm_start.get_union_of_double_stars_feasible_solution(A, remove_redundant_rows=True, remove_rows_with_zero_weight=True, choose_double_stars_greedily=True)
        double_greedy.append(len(p))
    print(single_non_greedy)
    print(single_greedy)
    print(double_non_greedy)
    print(double_greedy)
    # n = 15
    #
    # # Create an adjacency matrix for P_15
    # A = np.zeros((n, n), dtype=int)
    # for i in range(n - 1):
    #     A[i, i + 1] = 1
    #     A[i + 1, i] = 1  # Ensure symmetry

    # warm_start = WarmStart()
    # # print(warm_start.remove_redundant_rows([[1, 1, 1, 1], [1, 2, 3, 4], [1, 2, 3, 4], [1, 1, 1, 1]], [1, 2, 3, 5]))
    # p, w = warm_start.get_union_of_stars_feasible_solution(A, remove_redundant_rows=True, remove_rows_with_zero_weight=True, choose_stars_greedily=True)
    # # p, w = warm_start.get_union_of_double_stars_feasible_solution(A, remove_redundant_rows=True, remove_rows_with_zero_weight=True)
    # # p_rows, w = warm_start.get_clique_rows([1, 2, 3, 4, 5, 6, 7], 10)
    # # p = np.zeros((0, 10))
    # # for row in p_rows:
    # #     p = np.vstack((p, row))
    # # p = np.vstack((p, [1 for i in range(10)]))
    # # w = np.diag(w)
    #
    # print("p", p)
    # print("w", w)
    # #
    # #
    # print("p^Twp")
    # print(p.T @ w @ p)
    # print("w size", len(w[0]))

