Task:
_____________________________________________________________________________________________________________________________________________________________________________________________________________
1. Implement the cumulative divrank algorithm from the paper: http://www-personal.umich.edu/~qmei/pub/kdd10-divrank.pdf
2. Write Unit test case
_____________________________________________________________________________________________________________________________________________________________________________________________________________
Cumulative DivRank algorithm that attempts to balance between node centrality and diversity is used to rank nodes in the network.
In this project, the Cumulative DivRank is implemented and utilized to Rank the extracted keywords from the text file. 

Here are the following steps :
1. Preprocess the text - tokenization, removing the stopwords, and the steaming
2. Create the input for the cumulative DivRank algorithm- Generate GRAPH
3. The Cumulative DivRank is applied to rank the extracted keywords.
______________________________________________________________________________________________________________________________________________________________________________________________
1. Prerequisites:
    Python 3.7
    pip install -r "requirements.txt"

2. The Project is distributed with the following structure.
    1. "dataset":          contains the dataset on which the model is tested.
    2. "Python code file": contains the python code through which you can execute this project.
             I. "cum_cdvrank_main.py":   This python file is the main script of this project. Execute this script to performance Keywords ranking using Cumulative DivRank Algorithm.
            II. "cumulative_divrank.py": Contains the python code for Cumulative DivRank algorithm.
	   III. "test_unit_divrank.py":  Unit test cases are written here. Run this file to performance unit test for this project.   
3. Run: 
       I). To rank the keywords from the text, run the following command:
            Run "python cum_cdvrank_main.py"
            Output: The cumulative DivRank output will be saved in the dataset directory with the name input file name+"_output.txt" file.

       II). To perform the unit test for this project, run the following command:
			Run  "py.test test_unit_divrank.py".
			Run  "pytest -v -o junit_family=xunit1 --cov=. --cov-report xml:tests/coverage.xml --junitxml=tests/nosetests.xml".
			Run  "coverage report -m".

       III). Output of the unit test cases:
		The four unit test cases are written to check the quality and coverage of the project, the results are as follows:
		i.) All test cases are PASSED
			test_unit_divrank.py::test_txt_to_keywords_normal_input 	PASSED                                                                                                                                              [ 25%]
			test_unit_divrank.py::test_txt_to_keywords_wrong_input 		PASSED                                                                                                                                               [ 50%]
			test_unit_divrank.py::test_txt_to_sentences_normal_input 	PASSED                                                                                                                                             [ 75%]
			test_unit_divrank.py::test_case_file_with_nocontent 		PASSED        [100%]

		ii.) coverage report 
			Name                    Stmts   Miss  Cover   Missing
			-----------------------------------------------------
			cum_cdvrank_main.py       117     14    88%   54, 84, 148-160
			cumulative_divrank.py      46      0   100%
			test_unit_divrank.py       40      0   100%
			-----------------------------------------------------
			TOTAL                     203     14    93%


       IV). Output for the keywords ranking would look like this:
    Div_Rank_Keyword  DR_values
0             inform   0.096705
1               task   0.054801
2             divers   0.031312
3               rank   0.025516
4            network   0.025277
5                top   0.018808
6            prestig   0.011384
7             method   0.011122
8               item   0.010603
9               mani   0.007685
10             empir   0.007603
11          research   0.007513
12             relat   0.007467
13            select   0.006492
14           divrank   0.006397
15            vertic   0.006285
16            result   0.006263
17            search   0.006164
18              walk   0.006114
19            vertex   0.005971
20            random   0.005827
21         recommend   0.005768
22           section   0.005656
23             relev   0.005444
24           seafood   0.005265
25              data   0.005179
26             figur   0.005171
27            variou   0.005145
28            measur   0.005122
29            number   0.005112
30           restaur   0.005038
31             mobil   0.004973
32          pagerank   0.004959
33            exampl   0.004670
34            applic   0.004667
35            public   0.004641
36           theoret   0.004639
37             solid   0.004634
38            attach   0.004607
39          prestigi   0.004586
40          committe   0.004584
41            situat   0.004474
42           process   0.004404
43            author   0.004397
44             optim   0.004393
45        preferenti   0.004365
46            better   0.004350
47            differ   0.004347
48           foundat   0.004346
49               urn   0.004313
