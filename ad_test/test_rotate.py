from sixd_to_rotmat_batch import csv_to_matrix, matrix_to_6d, sixd_to_matrix, geodesic_loss


a = csv_to_matrix("test.csv")
print(a)
b = matrix_to_6d(a)
print(b)
c = sixd_to_matrix(b)
print(c)
