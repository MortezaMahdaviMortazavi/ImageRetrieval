import torch
import torch.nn as nn

class mAPAt100(nn.Module):
    def __init__(self, num_predictions):
        super(mAPAt100, self).__init__()
        self.num_predictions = num_predictions

    def forward(self, relevances, precision_at_ranks):
        num_queries = relevances.size(0)
        num_index_images = relevances.size(1)

        # Initialize variables to store cumulative precision and average precision
        cumulative_precision = torch.zeros(num_queries)
        average_precision = torch.zeros(num_queries)

        # Loop through each query image
        for i in range(num_queries):
            # Calculate the reciprocal rank for the query
            reciprocal_rank = 0
            for j in range(min(self.num_predictions, self.num_predictions)):
                if relevances[i, precision_at_ranks[j]] == 1:
                    reciprocal_rank = 1 / (j + 1)
                    break

            # Calculate precision at each rank for the query
            precision_at_r = torch.zeros(self.num_predictions)
            for j in range(self.num_predictions):
                precision_at_r[j] = relevances[i, precision_at_ranks[j]].sum() / (j + 1)

            # Calculate the average precision for the query
            average_precision[i] = (relevances[i] * precision_at_r).sum() / num_index_images

            # Calculate the cumulative precision for the query
            cumulative_precision[i] = precision_at_ranks[:100].float().mean()

        # Calculate mAP@100 as the mean of average precisions for all queries
        mAP_at_100 = average_precision.mean()

        return mAP_at_100, cumulative_precision
