# Code for "Improving Fair Classification by Leveraging Unlabeled Data in Post-processing"

## Requirements

```bash
pip install -r requirements.txt
```

## Adult Dataset

```bash
mkdir adult
wget -P ./adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -P ./adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

> Adult dataset is provided by Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.



## Evaluation

Run the proposed method by the following code.

```bash
python main.py --metric dp --model lr --attr sex --repeat 10
```

More options can be found.

```bash
python main.py -h
```