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

        /**return result;   


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

class Solution {
public:
    int lengthOfLongestSubstring(string s) 
    {
        int ans = 0;

        unordered_map<char, int>mp;
        int i = 0, j = 0, n = s.size();
        while(j<n)
        {
            mp[s[j]]++;
            while(mp[s[j]] > 1) mp[s[i++]]--;
            ans = max(ans, j-i+1);
            j++;
        }
        return ans;
    }
};

// Brute Force:
               // 1.Merge Both Array
              // 2.Sort them
             // 3.Find Median
            // TIME COMPLEXITY: O(n)+O(nlogn)+O(n)
            // SPACE COMPLEXITY: O(1)
 
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
       // Initialization some neccessary variables
        vector<int>v;
        
        // store the array in the new array
        for(auto num:nums1)   // O(n1)
            v.push_back(num);
        
        for(auto num:nums2)  // O(n2)
            v.push_back(num);
        
        // Sort the array to find the median
        sort(v.begin(),v.end());  // O(nlogn)
        
        // Find the median and Return it
        int n=v.size();  // O(n)
        
        return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2.0;
    }
};
