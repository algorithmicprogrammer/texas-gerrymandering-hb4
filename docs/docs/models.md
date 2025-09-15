# Machine Learning Models
## Classification 
### Using Linear Regression for Classification with One-Hot Encoding
In our linear regression model, the dependent variable will be the congressional district assignment. The independent variables will be the district's racial demographics and party affiliation. If it is discovered upon comparing coefficients that racial minority demographics have massive weights independent of party affiliation, this will indicate negative racial gerrymandering.

##### Using Random Forest for Classification
The following models will be trained:
<ul>
  <li>Determining district assignment by compactness scores.</li>
  <li>Determining district assignment by compactness scores and party affiliation.</li>
  <li>Determining district assignment in each Congressional District by compactness scores, party affiliation, and racial demographics.</li>
</ul>
If adding racial demographics features to the model drastically improves the model's performance, that is indicative of gerrymandering.

### Metrics
| Model        | Accuracy |Precision |   Recall | F1| ROC AUC|PR AUC| MCC |Log Loss
| :------------ | :---------: | ----------: | ---:|---:|--:|--:|---:|---:|
| Linear Regression       |        |  | | | | | | |
| Random Forest |             |    | | | | | | |

# Clustering
## K-Means Clustering
With k-means clustering, we will partition congressional districts into k groups. If the clusters align most closely with racial composition, then that is indicative of race determining the groupings - and of negative racial gerrymandering. If clusters disappear when racial features are removed from the model, then that indicates that race is shaping the congressional districts and is thus evidence of negative racial gerrymandering.
### Metrics
| Model        | Accuracy |Precision |   Recall | F1| Silhouette Score|ARI
| :------------ | :---------: | ----------: | ---:|---:|--:|--:|
| K-Means Clustering      |        |  | | | | |

