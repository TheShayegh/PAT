## Persian Address Tracer

Persian Address Tracer (PAT) is an intelligent system for converting Persian text of address to postal code. If you are not aware, it is worth mentioning that due to the special conditions in the Persian language and also the lack of standards in addressing in Iran, this is a difficult issue. As far as we know, the PAT system is the most successful and intelligent system available for this issue.

### Performance

Given the "lack of data" concern in Persian text processing problems, the model is trained and evaluated in the lack of data. Vividly by feeding more data to the model, performance will increase. There are two versions of model; One optimises the response time and the other is maximizing prediction accuracy.
Here is a brief report of the performance of the model, evaluated in the lack of data.
Note that in the following tables, for the complete addresses, the "Avenue Level Performance" is equal to the "Postal-Code Level Performance".

<br/>
<table>
  <caption>Response Accuracy Optimizer Model Performance (Response Speed: 5 it/s)</caption>
  <tr>
    <th rowspan="2">Number of Model Suggestions</th>
    <th colspan="3">Prediction Accuracy</th>
  </tr>
  <tr>
    <th>Parish Level</th>
    <th>Neighborhood Level</th>
    <th>Avenue Level</th>
  </tr>
  <tr>
    <th>1</th>
    <td>94.44</td>
    <td>84.64</td>
    <td>81.33</td>
  </tr>
  <tr>
    <th>2</th>
    <td>96.91</td>
    <td>89.15</td>
    <td>85.78</td>
  </tr>
  <tr>
    <th>3</th>
    <td>97.83</td>
    <td>91.17</td>
    <td>87.96</td>
  </tr>
  <tr>
    <th>5</th>
    <td>98.1</td>
    <td>92.85</td>
    <td>89.76</td>
  </tr>
  <tr>
    <th>10</th>
    <td>98.26</td>
    <td>94.71</td>
    <td>91.83</td>
  </tr>
</table>
<br/>
<table>
  <caption>Response Time Optimizer Model Performance (Response Speed: 41 it/s)</caption>
  <tr>
    <th rowspan="2">Number of Model Suggestions</th>
    <th colspan="3">Prediction Accuracy</th>
  </tr>
  <tr>
    <th>Parish Level</th>
    <th>Neighborhood Level</th>
    <th>Avenue Level</th>
  </tr>
  <tr>
    <th>1</th>
    <td>94.01</td>
    <td>82.43</td>
    <td>78.2</td>
  </tr>
  <tr>
    <th>2</th>
    <td>96.76</td>
    <td>88.92</td>
    <td>85.41</td>
  </tr>
  <tr>
    <th>3</th>
    <td>97.81</td>
    <td>91.28</td>
    <td>88.21</td>
  </tr>
  <tr>
    <th>5</th>
    <td>98.15</td>
    <td>93.26</td>
    <td>90.64</td>
  </tr>
  <tr>
    <th>10</th>
    <td>98.25</td>
    <td>94.94</td>
    <td>92.44</td>
  </tr>
</table>
<br/>

### Bargain

Currently, the model is trained and prepared just for Tehran addresses and doesn't support other cities. But it can be easily prepared for otherwheres just if there is suitable datasets (pairs of address and location). Performance depends on amonts of data but response time is independent.

You can order each model via my [email portal](mailto:behzad.shayegh.b@gmail.com). You can buy trained model or build it given your own dataset independently. Feel free to ask questions.

### Preprocessor Model will be Available for Free

In order to speed up research related to Persian language, the source code of the preprocessor model of this system will be available for free.

This preprocessor works unsupervised and does not require any cleaned or special data. All you need to do is give it plenty of unprocessed data. This model uses heuristic methods based on data statistics, therefore, as the volume of input data increases, the performance quality of the model also increases. It should be noted that this model does not use grammatical rules and therefore can be used in a wider range. Because it's unsupervised, there is a possibility of error in it's process, but it successfully standardize and reduce the number of types of input text. This model will work if you need to start your research quickly and temporarily go through the preprocessing stage. Look at these statistics about it's performance:

<table>
  <caption>Preprocessor Model Performance</caption>
  <tr>
    <th></th>
    <th>Before Preprocess</th>
    <th>After Preprocess</th>
  </tr>
  <tr>
    <th>Words Count</th>
    <td>21,712,597</td>
    <td>23,063,900</td>
  </tr>
  <tr>
    <th>Types Count</th>
    <td>85,108</td>
    <td>10,982</td>
  </tr>
</table>

You can get the preprocessor model by contacting me via my [email portal](mailto:behzad.shayegh.b@gmail.com). Feel free to ask questions.
