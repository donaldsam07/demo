import numpy as np

def create_train_data () :
    data = np.array(
        [['Sunny', 'Hot', 'High', 'Weak', 'no'],
         ['Sunny', 'Hot', 'High', 'Strong', 'no'],
         ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
         ['Rain', 'Mild', 'High', 'Weak', 'yes'],
         ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
         ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
         ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
         ['Overcast', 'Mild', 'High', 'Weak', 'no'],
         ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
         ['Rain', 'Mild', 'Normal', 'Weak', 'yes']]
    )
    return np . array ( data )
# train_data = create_train_data ()
# print ( train_data )

def compute_prior_probablity ( train_data ) :
    y_unique = ['no', 'yes']
    prior_probability = np . zeros (len ( y_unique ) )
    # Duyệt qua cột cuối cùng của dữ liệu (cột PlayTennis)
    y_column = train_data[:, -1]  # Cột cuối cùng là nhãn (PlayTennis)
    # Tính xác suất tiên nghiệm cho mỗi lớp
    for i, label in enumerate(y_unique):
        prior_probability[i] += np.sum(y_column == label) / len(train_data)
    return prior_probability
# prior_probablity = compute_prior_probablity ( train_data )
# print (("P( play tennis = No") , prior_probablity [0])
# print (("P( play tennis = Yes") , prior_probablity [1])


def compute_conditional_probability ( train_data ) :
    y_unique = ['no' , 'yes' ]
    conditional_probability = []
    list_x_name = []

    for i in range (0 , train_data . shape [1] -1) :
        x_unique = np . unique ( train_data[: , i ]) 
        list_x_name . append ( x_unique )
        #chứa các trường hợp của 1 cột
        x_conditional_probability = []
        for x_value in x_unique:
            probabilities = []
            
            # Duyệt qua từng nhãn (yes/no)
            for y_value in y_unique:
                # Lọc các hàng có nhãn là y_value
                rows_with_y = train_data[train_data[:, -1] == y_value]
                # Tính xác suất có điều kiện của x_value khi nhãn là y_value
                prob = np.sum(rows_with_y[:, i] == x_value) / len(rows_with_y)
                probabilities.append(prob)
            x_conditional_probability.append(probabilities)

    #xác xuất của điều kiện trong  bài
        conditional_probability . append ( x_conditional_probability )
    return conditional_probability , list_x_name
train_data = create_train_data ()
print(compute_conditional_probability(train_data))

# This function is used to return the index of the feature name
def get_index_from_value ( feature_name , list_features ):
    feature_name = feature_name.strip()  # Loại bỏ khoảng trắng thừa
    return np. where ( list_features == feature_name )[0][0]
# train_data = create_train_data ()
# conditional_probability , list_x_name = compute_conditional_probability ( train_data )
# # Compute P(" Outlook "=" Sunny "| Play Tennis "=" Yes ")
# x1= get_index_from_value ("Sunny", list_x_name [0])
# print ("P(' Outlook '=' Sunny '| Play Tennis '='Yes ') = ", np. round ( conditional_probability[0][x1][1] ,2))
# print ("P(' Outlook '=' Sunny '| Play Tennis '='No ') = ", np. round ( conditional_probability[0][x1][0] ,2))

'''
# train_data = create_train_data ()
# _, list_x_name = compute_conditional_probability ( train_data )
# outlook = list_x_name [0]
# #sắp xếp theo bảng chữ cái O->R->S
# i1 = get_index_from_value ("Overcast", outlook )
# i2 = get_index_from_value ("Rain", outlook )
# i3 = get_index_from_value ("Sunny", outlook )
# print (i1 , i2 , i3)
'''

def train_naive_bayes ( train_data ):
# Step 1: Calculate Prior Probability
    y_unique = ['no ', 'yes ']
    prior_probability = compute_prior_probablity ( train_data )

# Step 2: Calculate Conditional Probability
    conditional_probability , list_x_name = compute_conditional_probability ( train_data)

    return prior_probability , conditional_probability , list_x_name


def prediction_play_tennis (X , list_x_name , prior_probability , conditional_probability ):

    x1 = get_index_from_value ( X [0] , list_x_name [0])    # lấy vị trí của Sunny trong outlook
    x2 = get_index_from_value ( X [1] , list_x_name [1])    # lấy vị trí của Cool trong temperature
    x3 = get_index_from_value ( X [2] , list_x_name [2])    # lấy vị trí của high trong humidity
    x4 = get_index_from_value ( X [3] , list_x_name [3])    # lấy vị trí của srtong trong wind

    p0 = 0  # ti le 'no'
    p1 = 0  # ti le 'yes'

#   p0= P_yes* P(Sunny|no)* P(Cool|no)* P(High|no)* P(Strong|no)
#   p1= P_yes* P(Sunny|yes)* P(Cool|yes)* P(High|yes)* P(Strong|yes)

    p0 = prior_probability[0] * \
        conditional_probability[0][x1][0] *\
        conditional_probability[1][x2][0] *\
        conditional_probability[2][x3][0] *\
        conditional_probability[3][x4][0]

    p1 = prior_probability[1] * \
        conditional_probability[0][x1][1] *\
        conditional_probability[1][x2][1] *\
        conditional_probability[2][x3][1] *\
        conditional_probability[3][x4][1]

    if p0 > p1 :
        y_pred =0
    else :
        y_pred =1

    return y_pred

X = ['Sunny ','Cool ', 'High ', 'Strong ']
data = create_train_data ()
prior_probability , conditional_probability , list_x_name = train_naive_bayes ( data )
pred = prediction_play_tennis (X, list_x_name , prior_probability ,conditional_probability )

if( pred ):
    print ("Ad should go!")
else :
    print ("Ad should not go!")
