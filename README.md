class Solution {
public:
    vector<int> minOperations(string boxes) {
        int n = boxes.size();
        vector<int> left(n, 0), right(n, 0), res(n, 0);
        int count = (boxes.at(0) == '1' ? 1 : 0);

        for (int i = 1; i < n; i++) {
            left[i] = left[i - 1] + count;
            count += (boxes.at(i) == '1' ? 1 : 0);
        }

        count = (boxes.at(n-1) == '1' ? 1 : 0);

        for (int i = n - 2; i >= 0; i--) {
            right[i] = right[i + 1] + count;
            count += (boxes.at(i) == '1' ? 1 : 0);
        }

        for (int i = 0; i < n; i++) {
            res[i] = left[i] + right[i];
        }

        /**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* twoSum(int* nums, int numsSize, int target, int* returnSize) {
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

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* dummyHead = new ListNode(0);
        ListNode* tail = dummyHead;
        int carry = 0;

        while (l1 != nullptr || l2 != nullptr || carry != 0) {
            int digit1 = (l1 != nullptr) ? l1->val : 0;
            int digit2 = (l2 != nullptr) ? l2->val : 0;

            int sum = digit1 + digit2 + carry;
            int digit = sum % 10;
            carry = sum / 10;

            ListNode* newNode = new ListNode(digit);
            tail->next = newNode;
            tail = tail->next;

            l1 = (l1 != nullptr) ? l1->next : nullptr;
            l2 = (l2 != nullptr) ? l2->next : nullptr;
        }

        ListNode* result = dummyHead->next;
        delete dummyHead;
        return result;
    }
};
