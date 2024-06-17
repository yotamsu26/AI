314055153
206922205
*****
Comments:
Heuristic: Our heuristic function is a linear combination of a few aspect, all of them
are fulfilling the idea of the strategy that keeps the board as much closer to a "snake" like shape, where the
highest number is at the corner and then slowly decreasing in a shape that seems like snake:
- max tile score
    the value of the max tile
- number of empty tiles
    how many empty tiles are there
- merging potential - how many tiles can be merged
    For each 2 blocks that could be merged together add the sum of those 2 tiles. (for example if there is 2 128 tiles
    that are neighbors then add 2*128 to the potential.
- monotonicity in the rows and cols
    Check Monotonicity in the cols and rows, for each row and each col, if it is monotonic (increasing or decreasing)
    than add the sum of the tiles in that row/col - we want to enforce the "snake" positioning so
    the first and third row should be decreasing and the second and forth rows should be increasing.
    all the cols need to be decreasing
    return the sum of all the monotonic rows and cols.
- smoothness of the neighbors values
    calculate the diff between each tiles and its 2 neighbors - we need that diff to be as minimal as possible.
    (similar reason as monotonicity)
    this score decrease the value of the node as higher diffs are not good (near 512 should be 256 and not 2)
- tile alignment to a shape of "snake" matrix
    we wanted to achive this positioning (where 16 is the highest tile and 1 is the smallest)
    [
        [16, 15, 14, 13],
        [9, 10, 11, 12],
        [8, 7, 6, 5],
        [1, 2, 3, 4]
    ]
- corner bias (max tile on the corner)
    Add a bias that the max tile would be in the 0,0 corner

we also tried to compute the heuristic on all 4 rotations of the board as it doesn't matter that we
chose the top left corner as our reference corner, but it really slowed down and didn't improve a lot the results

The weights of the linear combination decided based on a trial and error while thinking what metrics more important and
What are the scale of each metric.