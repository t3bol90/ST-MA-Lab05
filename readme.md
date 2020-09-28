# Lab05:  Mô Hình Hồi Quy Tuyến Tính
## Phân lớp văn bản với kĩ thuật bình phương tối tiểu
Courses `MTH00051`: Toán ứng dụng và thống kê
`18CLC6`, `FIT - HCMUS`.
`11/09/2020`

Đây là đồ án cá nhân, do một thành viên thực hiện:
-   `18127231`: Đoàn Đình Toàn (GitHub: [@t3bol90](https://github.com/t3bol90))



```python
# DON'T CHANGE this part: import libraries
import numpy as np
import scipy
import json
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import re
import itertools
```


```python
# DON'T CHANGE this part: read data path
train_set_path, valid_set_path, random_number = input().split()
```

     train.json valid.json 69


## 1. Xử lý dữ liệu dạng văn bản:

Đầu tiên ta cần đọc data vào, sau khi sử dụng json để load file lên thì ta tách ra hai cột cần thiết cho bài toán là cột 'reviewText' và cột 'overall' (đối với cả tập train và tập valid).


```python
def load_data(path):
    """
    load_data split 'reviewText', 'overall' as values list for training/validation.
    :param path: directory of json file.
    :return text: list of string as document.
    :return overall: label for each document's overall.
    """
    test = json.load(open(path))
    texts = [element['reviewText'] for element in test]
    overall = [element['overall'] for element in test]
    return texts, overall
```


```python
train_text, train_overall = load_data(train_set_path)
test_text, test_overall   = load_data(valid_set_path)
```

Sơ lược qua thì các bước cần làm theo thứ tự là: [convert number], [tokenizer], [stopwords], [stemming]. Tiền xử lí lại chia làm 2 phần cho hai giai đoạn là train và valid nên các phần sau đây sẽ trình bày từ lúc tiền xử lí cho tới lúc ra được các vector histogram.

Đầu tiên để phục vụ cho phần đầu tiên của tiền xử lí là loại bỏ số ra khỏi văn bản, ban đầu em tính sử dụng `regex`, nhưng với sự mu muội ban đầu thì regex đã làm quá tốt phần của mình (với pattern`^[-+]?\d+$`) để trim tất cả các số nguyên ra khỏi chuỗi. Nhưng sau cùng thì nhận ra là yêu cầu đơn giản hơn thế rất nhiều, ta chỉ cần chuyển các word là số về 'num'. Do đó 2 hàm bên dưới viết ra với mục đích kiểm tra xem chuỗi nhập vào có phải là số hay không. (Nguồn: Loot từ [stackoverflow](https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except/1267145#1267145))


```python
def RepresentsInt(s):
    """
    RepresentsInt check if a string is a integer or not.
    :param s: string
    :return: is_integer?
    """
    try: 
        int(s)
        return True
    except ValueError:
        return False
def RepresentsFloat(s):
    """
    RepresentsFloat check if a string is a float or not.
    :param s: string
    :return: is_float?
    """
    try: 
        float(s)
        return True
    except ValueError:
        return False
```

Để thực hiện tiền xử lí, từ được đưa qua các bước như sau:

- Chuyển thành chữ in thường:
    Sử dụng method lower() của str.
- Chuyển chuỗi số thành 'num':
    Sử dụng hai hàm `RepresentsInt` và `RepresentsFloat` đã được viết ở trên, ta có thể thực hiện forloop qua các word của chuỗi và chuyển đổi chung thành 'num'.
- Phân tách thành các thành phần:
    Sử dụng hàm word_tokenize của nltk.
- Loại bỏ các stopwords như a, an, the...:
    Sử dụng list stopwords của tiếng Anh từ thư viện nltk, ta thực hiện forloop qua các word của chuỗi và chỉ giữ lại các từ không nằm trong stopwords list.
- Stemming:
    Sử dụng PorterStemmer của nltk làm stemmer, tiến hành stem lần lượt các từ trong document.

[!] Lưu ý:
- Ở đây stopWords và stemmer được đặt ở global vì bản chất của chúng không thay đổi, nên việc này có thể cải thiện hiệu suất của quá trình preprocessing lên rất nhiều. (Cụ thể performance hiện tại là 3.8s cho tất cả test và train data, trong khi nếu ta đặt vào thì bước này có thể lên tới 4xs).
- Bộ stopwords của nltk cần phải tải về, nếu chạy ra lỗi chưa tải bộ stopword, vui lòng sử dụng dòng lệnh sau (vì đề không cho import thêm thư viện nếu không thì cũng xử lí ở đây luôn cũng được).
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```


```python
stopWords = set(stopwords.words('english')) # stopword dictionary for english
stemmer = PorterStemmer()                   # stemmer for english

def preprocess(text, state=False):
    """
    :preprocess: converting text to lowercase, coverting number, tokenization, removing stopword, stemming.
    :param: text: str for preprocessing - it can be from train/test data.
    :param: state: is preprocessing for word set (return a set) or document embedding (return a list).
    :return: state=False -> set - bag-of-word of text after preprocess.
             state=True -> list - list word of text after preprocess.
    """
    # converting text to lowercase
    text = text.lower()
    
    # converting number -> 'num'
    numalpha = []
    for t in text.split():
        if RepresentsInt(t) or RepresentsFloat(t):
            numalpha.append('num')
        else:
            numalpha.append(t)
    text = ' '.join(numalpha)
    
    # tokenization
    text = word_tokenize(text)
    
    # remove stopwords
    words_filter = []
    for w in text:
        if w not in stopWords:
            words_filter.append(w)
    result = []
    
    # stemming 
    if not state:
        stemmed_word = set()
        for w in words_filter:
            stemmed_word.add(stemmer.stem(w))
        result = stemmed_word
    if state:
        stemmed_word = []
        for w in words_filter:
            stemmed_word.append(stemmer.stem(w))
        result = stemmed_word
        
    return result
```

Kế tiếp, ta cần dựng lên vocab hay bag-of-word, từ điển đối với các từ của tập train. Tiến hành preprocess đối với từng document trong tập train sau đó đổ dồn chúng vào một set ta sẽ có được list vocabulary.


```python
def get_setdata(text_list):
    """
    get_setdata return a bag-of-word of train data.
    :param text_list: train list of str as documents.
    :return: a set as bag-of-word.
    """
    word_set = set()
    for text in text_list:
        word_set = word_set|preprocess(text)
    return word_set
```

Sau khi tạo được word_set cho tập train, ta tiến hành thêm từ 'unk' vào và sau đó sort lại word_set với mục đích là giữ một thứ tự đồng nhất khi chuyển đổi giữa các trường/tập dữ liệu với nhau. Vì cơ bản set và dict của python implement bên dưới theo phương pháp hashing nên sẽ không giữ được thứ tự khi chuyển đổi, dẫn đến khó xử lí cho bước build ra histogram vector. Để dạng list và có thứ tự sort vẫn là một giải pháp có thể chấp nhận được.


```python
word_set = get_setdata(train_text)
word_set.add('unk')
word_set = sorted(word_set)
```

Kế tiếp, ta tiến hành 'nhúng' các document của tập train vào trong word_set để tạo ra các histogram vector. Hàm `document_embedding` bên dưới có khả năng trả ra histogram vector của một document, với n documents từ tập train, ta chạy vòng for n lần để có được tất cả các histogram vector của cả tập.

Sơ qua về histogram vector, ở đây đơn giản chỉ là vector tần suất của các word xuất hiện trong document dựa trên tập vocab. Vector tần suất này được chuẩn hóa bằng cách chia lại cho một scalar là tổng của các giá trị trong vector để shift về khoảng [0,1].


```python
def document_embedding(vocab_set, document):
    """
    document_embedding return a histogram vector of train document
    :param vocab_set: a set as bag-of-word for document embeding
    :param document: a str represent document - review from user
    :return: a histogram vector (m,1) with m is size of vocab_set and sum = 1
    """
    vocab_list = {word: 0 for word in vocab_set}
    doc_set = preprocess(document,True)
    for word in doc_set:
        vocab_list[word] += 1
    vocab_list = np.array(list(vocab_list.values()))
    return np.divide(vocab_list,sum(vocab_list))
```


```python
# Get histogram vector from all input
historgrams = []
for doc in train_text:
    vector = document_embedding(word_set,doc)
    historgrams.append(vector)
```

Ta cũng phải nhúng các document của tập valid vào trong word_set, vì các document của tập valid có thể có chứa các ký tự lạ nên bước chuyển các kí tự lạ cũng được thực hiện ở đây. `document_inference` trả về một histogram vector từ tập valid và thêm một doc_set chính là list từ của document trong tập valid sau khi preprocessing.


```python
def document_inference(vocab_set, document):
    """
    document_inference return a histogram vector of validate document.
    :param vocab_set: a set as bag-of-word for document embeding.
    :param document: a str represent document - review from user.
    :return vector: a histogram vector (m,1) with m is size of vocab_set and sum = 1.
    :return doc_set: a list of document's words after preprocessing.
    """
    vocab_list = {word: 0 for word in vocab_set}
    doc_set = preprocess(document,True)
    for i in range(len(doc_set)):
        if doc_set[i] not in vocab_set:
            doc_set[i] = 'unk'
        vocab_list[doc_set[i]] += 1
    vocab_list = np.array(list(vocab_list.values()))
    return np.divide(vocab_list,sum(vocab_list)), doc_set
```


```python
# Get histogram vector from all valid data
historgrams_valid = []
preprocessing_docs = []
for doc in test_text:
    vector, doc_set = document_inference(word_set,doc)
    historgrams_valid.append(vector)
    preprocessing_docs.append(doc_set)
```

In ra đoạn sau khi preprocess theo con số random đọc từ người chấm bài.


```python
print(preprocessing_docs[int(random_number)])
```

    ['must', 'tool', 'needl', 'felt', '!', 'great', 'qualiti', 'work', 'wonder', '!', 'real', 'time', 'saver', '!']


## 2. Sử dụng mô hình hồi quy tuyến tính dùng bình phương tối tiểu

Để xây dựng mô hình $y = ax + b$ hay $y = \theta_{0} + \theta_{1}x$. Ta tiến hành lập ma trận A và ma trận b sao cho:
$$
Ax = b
$$


```python
def get_model(historgrams, predict):
    """
    get_model return matrix A (n,m+1) as a weight matrix for regression and b as predicted label of training documents.
    :param histograms: a list of histogram vector from trainning documents.
    :param predict: label of training documents (rating/overall).
    :return: A (n,m+1), b(n,5) with 5 is hardcoded as number of value in labels
    """
    col1 = np.ones((len(predict),1))
    colx = np.array(historgrams)
    A = np.hstack((col1,colx))
    b = np.zeros((len(predict),5))
    for i in range(len(predict)):
        b[i][int(predict[i]-1)] = 1.0
    return A, b
```

Xây dựng hàm `get_model` trả về ma trận $A$, và ma trận $b$. Ma trận $A$ có được bằng cách ghép cột (n,1) toàn số 1 vào các histograms của tập train. Do đó ma trận $A$ sẽ có chiều (n,m+1). Kế đến ma trận $b$, ma trận $b$ có được bằng cách chuẩn hóa label tương ứng với index của label đó trong vector xác suất của document đó trong ma trận $b$. Giả sử ta có label của document thứ i là 5 thì b[i] = [0,0,0,0,1]. Như vậy chiều của $b$ là (n,5).


```python
def get_A_valid(historgrams):
    """
    get_A_valid return matrix A (n,m+1) as a weight matrix for regression.
    :param histograms: a list of histogram vector from validate documents.
    :return: matrix A (n,m+1)
    """
    col1 = np.ones((len(historgrams),1))
    colx = np.array(historgrams)
    A = np.hstack((col1,colx))
    return A
```

Ta cũng xây dựng một ma trận $A$ tương ứng cho tập valid.


```python
A, b = get_model(historgrams, train_overall)
```


```python
A_valid = get_A_valid(historgrams_valid)
```

$$
\hat{x} = A^\dagger.b
$$
Do $A$ có chiều là (n,m+1), pseudo invese của $A$ hay $A^\dagger$ có số chiều là (m+1,n) và $b$ có chiều là (n,5) nên ta sẽ có $\hat{x}$ sẽ có số chiều là (m+1,5).
*Không hiểu sao nhưng tính cái pinv(A) trên Ubuntu rất chậm. Chuyển sang máy của bạn mình xài Windows thì lại chỉ còn 52s. Trong khi mình time tầm ~4min.*


```python
x_hat = np.linalg.pinv(A) @ b
```

Sau đó tiến hành fit vào tập valid:
$$
y_{\text{predict}} = A_\text{valid}.\hat{x}
$$

Trong đó thì do $\hat{x}$ có số chiều là (m+1,5) tương ứng với $\theta_{0}$ có chiều (1,5) và $\theta_{1}$ có chiều (m,5). $A_\text{valid}$ có chiều (n,m+1) do đó $y_{\text{predict}}$ sẽ có chiều (n,5). Mỗi dòng của $y_{\text{predict}}$ là một vector tương ứng với label predicted.


```python
y_pred = A_valid @ x_hat
```

## 3. Sử dụng độ chính xác để đánh giá mô hình

Đưa y_predict qua lớp softmax, ta sẽ có được label tương ứng với từng document. Ở đây lưu ý là ta +1 vì ở trên lúc tiền xử lí ta có -1 của cột predict đi để chuẩn hóa thành index của mảng (index đếm từ 0).


```python
y_label = np.argmax(scipy.special.softmax(y_pred, axis = 1), axis = 1) + 1
```

Tính accuracy bằng cách lấy $y_\text{label}$ trừ đi $y_\text{ground truth}$ sau đó đếm các ô không phải là 0 (nghĩa là các ô không đúng) sau đó chia cho tổng số document trong tập valid. Bằng cách này, ta có được tỉ lệ lỗi trên tập label. Lấy 1 trừ đi tỉ lệ này sẽ ra accuracy của model.


```python
arc = 1 - np.count_nonzero(y_label - np.array(test_overall,dtype=np.int8))/len(test_overall)
```

Tèn tenn, in ra kết quả hoy.


```python
print(f'M2 - {arc}')
```

    M2 - 0.506


## 4. Kết luận:

Mô hình trên được xây dựng và tuân theo yêu cầu đề bài của bài thực hành này. Điều này đồng nghĩa với việc không có thêm các thao tác khác khi preprecessing (các kỹ thuật của NLP). Ở model chỉ dùng model $y = ax + b$ chứ không dùng model khác. Do đó, accuracy có thể chấp nhận được với model này.

Bài lab này khá thú vị vì việc không sử dung thư viện skitlearn đã giúp chúng em tính chiều của ma trận một cách kỹ càng trước khi thực hiện phép nhân. Một lần nữa, em cảm thấy như mình tiến sâu hơn và hiểu hơn về model mà mình vẫn thường gọi một dòng lệnh trong skitlearn nó làm như thế nào.


Chân thành cảm ơn đội ngũ TA, đặc biệt là cô Trần Thị Thảo Nhi đã không quản ngày đêm reply mail của em. Cảm ơn giảng viên giảng dạy bộ môn, thầy Bùi Huy Thông. Hẹn gặp mọi người ở một môn khác.
