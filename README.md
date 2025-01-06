# anushya
int* result = (int*)malloc(2 * sizeof(int));
    *returnSize = 2;
    
    for (int i = 0; i < numsSize; i++) {
        for (int j = i + 1; j < numsSize; j++) {
            if (nums[i] + nums[j] == target) {
                result[0] = i;
                result[1] = j;
                return result;
            }
        }
    }
    
    result[0] = 0;
    result[1] = 1;
    return result;
}
Input Example

public class Solution {
    public bool IsPalindrome(int x) {
        string y = x.ToString();
        if(y.Length == 1)
        {
            return true;  
        }
        int i = 0;
            for(int j = y.Length-1; j <= y.Length-1; j--)
            {
                if(y[j]==y[i])
                {
                    i++;
                    if(i == y.Length-1)
                    {
                        return true;
                    }
                    continue;
                }
                else
                {
                    return false;
                }
            }
            return true;          
        return false; 
    }
}
