from pyspark import SparkContext, SparkFiles
import itertools
import os

# André Luis - 62058
# Clony Abreu - 94085

#todo change code to allow the user information to indicate the file to be read
FILE_NAME_TO_READ = os.path.join("data", "soc-LiveJournal1Adj_sample.txt")
DIRECTORY_NAME_TO_SAVE = os.path.join("data", "ASSIGNMENT1-1")

class MDLEAssignment1(object):

    @staticmethod
    def parse_to_bonded_friendship(line):
        """
        This method parses a line from the text file into a "bonded friendship" structure.
        For example, the line: ``0    1,2,3,4, ...`` will get parsed to:
        ``(0, [1, 2, 3, 4, ...])`` wich will allow create new friendship in
        That is, a python tuple where the first element is an int and the second element is a List[int]
        :param line: str
        :return: Tuple[int, List[int]], the parsed line
        """
        split = line.split()
        user_id = int(split[0])

        if len(split) == 1:
            friends = []
        else:
            friends = list(map(lambda x: int(x), split[1].split(',')))

        return user_id, friends

    @staticmethod
    def bonded_friendship_to_connect(bonded_friendships):
        """
        This method maps a " bonded friendship" structure (see the above method) to an array of connections.
        For example, the value ``(0, [1, 2, 3])`` will get mapped to::
            [
                ((0,1), 0),
                ((0,2), 0),
                ((0,3), 0),
                ((1,2), 1),
                ((1,3), 1),
                ((2,3), 1)
            ]
        Essentially, the friend ownership structure is converted a list of all connection information embedded in the
        structure. For example, users 0 and 1 are already connected, so that connection is represented by ``((0,1), 0)``.
        The final ``0`` indicates that these users are currently friends.
        As another example, the structure encodes the fact that users 1 and 2 have a common friend (in this case, the mutual
        friend is friend 0). So, the resulting connection is represented by ``((1,2), 1)``, where the ``1`` indicates that
        these users share a single common friend.
        Finally, it is important to note that the "key" in each of these elements (namely the ``(user_id_0, user_id_1)``
        pair) is deterministically ordered. It is important for each unique pair of users to be grouped in the same way,
        so the bi-directional relationship must be retained by ordering the tuple by userId (or any other deterministic
        ordering. Simple "greater-than" comparison just happens to be the fastest).
        :param bonded_friendships: the friendship bonded object
        :return: List[Tuple[Tuple[int, int], int]] the embedded connections
        """
        user_id = bonded_friendships[0]
        friends = bonded_friendships[1]

        connections = []

        for friend_id in friends:
            key = (user_id, friend_id)
            if user_id > friend_id:
                key = (friend_id, user_id)

            connections.append(
                (key, 0)
            )

        for friend_pair in itertools.combinations(friends, 2):
            friend_0 = friend_pair[0]
            friend_1 = friend_pair[1]

            key = (friend_0, friend_1)
            if friend_0 > friend_1:
                key = (friend_1, friend_0)
            connections.append(
                (key, 1)
            )

        return connections

    @staticmethod
    def score_of_common_friends_to_suggests(common_friend_score_item):
        """
        Maps a "common friend score" object to two distinct suggestions. The value
        ``((0, 1), 21)`` encodes that users 0 and 1 share 21 common friends. This means that user 1 should be suggested
        to user 0 AND that user 0 should be suggested to user 1. For every input to this function, two "suggestions"
        will be returned in a List.
        A "recommendation" has the following form::
            (user_id_0, (suggested_user, common_friends_score))
        :param common_friend_score_item: a common friend score item
        :return: List[Tuple[int, Tuple[int, int]]] two recommendation items
        """
        connection = common_friend_score_item[0]
        score = common_friend_score_item[1]

        friend_0 = connection[0]
        friend_1 = connection[1]

        suggestion_0 = (friend_0, (friend_1, score))
        suggestion_1 = (friend_1, (friend_0, score))

        return [suggestion_0, suggestion_1]

    @staticmethod
    def suggestions_to_ordain(suggestions):
        if len(suggestions) > 1024:
            # First find the top elements in suggestions (if log(len(suggestions)) > 10), then ordain it.
            # This works as an optimization to run in O(n), where n is the length of suggestions.
            max_indexes = []

            current_max_index = 0
            for i in range(1, len(suggestions)):
                suggest = suggestions[i]
                if suggest[1] >= suggestions[current_max_index][1] and i not in max_indexes:
                    current_max_index = i

            max_indexes.append(current_max_index)
            suggestions = [suggestions[i] for i in max_indexes]

        # Ordain by common friend, then by user_id (for equal number of common friends between users)
        suggestions.sort(key=lambda x: (-x[1], x[0]))

        # Map every [(user_id, common_score), ...] to [user_id, ...]
        return list(map(lambda x: x[0], suggestions))

if __name__ == '__main__':
    # Initialize spark context
    sc = SparkContext(appName="Spark_FileReader")

    # Read from text file, split each line into "words" by any whitespace (i.e. empty parameters to string.split())
    lines = sc.textFile(FILE_NAME_TO_READ)

    # Map each line to the form: (user_id, [friend_id_0, friend_id_1, ...])
    friendship_bind = lines.map(MDLEAssignment1().parse_to_bonded_friendship)

    # Map each "friendship_bind" to multiple instances of ((user_id, friend_id), VALUE).
    # VALUE = 0 indicates that user_id and friend_id are already friends.
    # VALUE = 1 indicates that user_id and friend_id are not friends.
    friend_border = friendship_bind.flatMap(MDLEAssignment1().bonded_friendship_to_connect)
    friend_border.cache()

    # Filter all pairs of users that are already friends, then sum all the "1" values to get their common friend score.
    common_friend_score = friend_border.groupByKey() \
        .filter(lambda border: 0 not in border[1]) \
        .map(lambda border: (border[0], sum(border[1])))

    # Create the suggestions objects, group them by key, then ordain.
    suggestions = common_friend_score.flatMap(MDLEAssignment1().score_of_common_friends_to_suggests) \
        .groupByKey() \
        .map(lambda m: "{}\t{}".format(m[0], str(MDLEAssignment1().suggestions_to_ordain(list(m[1])))[1:-1]))

    # Save to output directory, end context
    suggestions.saveAsTextFile(DIRECTORY_NAME_TO_SAVE)
    sc.stop()