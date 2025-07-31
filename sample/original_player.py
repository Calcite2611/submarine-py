from submarine_py import Player, play_game, Ship, Reporter, Field
import json
import random
import logging
import numpy as np

notExist = -0.5  # 予測マップ上で存在しない位置を表す値

def report_moved(moved):
    """移動された場合の文章を作る．"""
    if moved["distance"][0] > 0:
        arrow = ">" * moved["distance"][0]
    elif moved["distance"][0] < 0:
        arrow = "<" * (-moved["distance"][0])
    elif moved["distance"][1] > 0:
        arrow = "v" * moved["distance"][1]
    elif moved["distance"][1] < 0:
        arrow = "^" * (-moved["distance"][1])
    print(f' moved {moved["ship"]} by {arrow}')


def report_attacked(attacked):    
    """
    攻撃した，された場合の文章を作る．
    """
    msg = f' attacked {attacked["position"]}'
    if "hit" in attacked:
        msg += " hit " + attacked["hit"]
    if "near" in attacked:
        msg += " near " + str(attacked["near"])
    print(msg)


def report_observation(observation):
    """
    相手の艦の状態を通知する．
    """
    msg = "opponent ships: "
    for type, state in observation["opponent"].items():
        msg += type + ":" + str(state["hp"]) + " "
    print(msg)


class OriginalPlayer(Player):
    def __init__(self):
        super().__init__()
        self.rng = random.Random(42 or None)
        self.enemy_w = None # 判明している敵のwの位置
        self.enemy_c = None # 判明している敵のcの位置
        self.enemy_s = None # 判明している敵のsの位置
        # 予測マップの初期化はinitializeメソッドで行う
        self.w_pos_pred = None
        self.c_pos_pred = None
        self.s_pos_pred = None
        self.entire_map = None

    def initialize(self, field):
        super().initialize(field)
        # フィールドが設定された後に予測マップを初期化
        self.w_pos_pred = np.zeros((self.field.width, self.field.height))
        self.c_pos_pred = np.zeros((self.field.width, self.field.height))
        self.s_pos_pred = np.zeros((self.field.width, self.field.height))
        self.entire_map = np.zeros((self.field.width, self.field.height))

    def name(self):
        return 'original-player'
    
    def place_ship(self):
        distance = 2  # 2マス以上離す
        placed_positions = set()
        ship_types = ['w', 'c', 's']
        max_attempts = 500
        for ship_type in ship_types:
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                attempts += 1

                # choice a random position which is not occupied, and scattered.
                row = self.rng.randint(0, self.field.height - 1)
                col = self.rng.randint(0, self.field.width - 1)
                position = [col, row]

                if tuple(position) not in placed_positions:
                    is_isolated = True
                    for placed_pos in placed_positions:
                        if abs(position[1] - placed_pos[1]) <= distance and abs(position[0] - placed_pos[0]) <= distance:
                            # If the new position is too close to any placed ship, break
                            is_isolated = False
                            break
                    
                    if is_isolated:
                        placed_positions.add(tuple(position))
                        placed = True
                        logging.info(f"Placed {ship_type} at {position}")
            if not placed:
                logging.warning(f"Failed to place {ship_type} after {max_attempts} attempts. Trying to place in a random way.")
                for r in range(self.field.height):
                    for c in range(self.field.width):
                        if (c, r) not in placed_positions:
                            placed_positions.add((c, r))
                            logging.info(f"Placed {ship_type} at {[c, r]}")
                            break
                    if placed:
                        break

        return {
            'w': list(list(placed_positions)[0]),
            'c': list(list(placed_positions)[1]),
            's': list(list(placed_positions)[2])
        }
    
    def predict_position_motion(self, moved):
        """
        相手の行動が移動であった場合の予測マップの更新を行う
        """
        if moved['ship'] == 'w':
            if self.enemy_w is not None:
                self.enemy_w[0] += moved['distance'][0]
                self.enemy_w[1] += moved['distance'][1]
            else:
                self.w_pos_pred = self.update_prediction_map_motion(self.w_pos_pred, moved['distance'])
        elif moved['ship'] == 'c':
            if self.enemy_c is not None:
                self.enemy_c[0] += moved['distance'][0]
                self.enemy_c[1] += moved['distance'][1]
            else:
                self.c_pos_pred = self.update_prediction_map_motion(self.c_pos_pred, moved['distance'])
        elif moved['ship'] == 's':
            if self.enemy_s is not None:
                self.enemy_s[0] += moved['distance'][0]
                self.enemy_s[1] += moved['distance'][1]
            else:
                self.s_pos_pred = self.update_prediction_map_motion(self.s_pos_pred, moved['distance'])
    
    def update_prediction_map_motion(self, map, motion):
        x = motion[0]
        y = motion[1]
        pred_map = np.full((self.field.width, self.field.height), notExist)
        for i in range(self.field.width):
            for j in range(self.field.height):
                if i + x >= 0 and i + x < self.field.width and j + y >= 0 and j + y < self.field.height:
                    pred_map[i + x][j + y] = map[i][j]
        return pred_map
    
    def predict_position_attack(self, attacked):
        """
        相手の行動が攻撃であった場合の予測マップの更新を行う
        """
        x = attacked['position'][0]
        y = attacked['position'][1]

        ## 攻撃が当たった場合の処理
        if 'hit' in attacked:
            if attacked['hit'] == 'w':
                if self.enemy_w == attacked['position'] or self.enemy_w is None:
                    self.w_pos_pred[x][y] = 1
                else:
                    print(f"Warning(Debug): enemy_w position {self.enemy_w} does not match attacked position {attacked['position']}")

            elif attacked['hit'] == 'c':
                if self.enemy_c == attacked['position'] or self.enemy_c is None:
                    self.c_pos_pred[x][y] = 1
                else:
                    print(f"Warning(Debug): enemy_c position {self.enemy_c} does not match attacked position {attacked['position']}")

            elif attacked['hit'] == 's':
                if self.enemy_s == attacked['position'] or self.enemy_s is None:
                    self.s_pos_pred[x][y] = 1
                else:
                    print(f"Warning(Debug): enemy_s position {self.enemy_s} does not match attacked position {attacked['position']}")

        ## 水飛沫が発生した場合の処理
        if 'near' in attacked:
            if attacked['near'] == 'w':
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if i == 0 and j == 0:
                            self.w_pos_pred[x + i][y + j] = notExist
                        if 0 <= x + i < self.field.width and 0 <= y + j < self.field.height:
                            if self.w_pos_pred[x + i][y + j] is not notExist:
                                self.w_pos_pred[x + i][y + j] =  1

            elif attacked['near'] == 'c':
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if i == 0 and j == 0:
                            self.c_pos_pred[x + i][y + j] = notExist
                        if 0 <= x + i < self.field.width and 0 <= y + j < self.field.height:
                            if self.c_pos_pred[x + i][y + j] is not notExist:
                                self.c_pos_pred[x + i][y + j] =  1

            elif attacked['near'] == 's':
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if i == 0 and j == 0:
                            self.s_pos_pred[x + i][y + j] = notExist
                        if 0 <= x + i < self.field.width and 0 <= y + j < self.field.height:
                            if self.s_pos_pred[x + i][y + j] is not notExist:
                                self.s_pos_pred[x + i][y + j] =  1

        ## 何も当たらなかった場合の処理
        if 'hit' not in attacked and 'near' not in attacked:
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if 0 <= x + i < self.field.width and 0 <= y + j < self.field.height:
                        if self.w_pos_pred[x + i][y + j] is not notExist:
                            self.w_pos_pred[x + i][y + j] = notExist
                        if self.c_pos_pred[x + i][y + j] is not notExist:
                            self.c_pos_pred[x + i][y + j] = notExist
                        if self.s_pos_pred[x + i][y + j] is not notExist:
                            self.s_pos_pred[x + i][y + j] = notExist

    def update_entire_map(self, observation):
        """
        全体の予測マップを更新する
        """
        self.entire_map = np.zeros((self.field.width, self.field.height))
        for type, state in observation['opponent'].items():
            if type == 'w' and state["hp"] > 0:
                self.entire_map += self.w_pos_pred
            elif type == 'c' and state["hp"] > 0:
                self.entire_map += self.c_pos_pred
            elif type == 's' and state["hp"] > 0:
                self.entire_map += self.s_pos_pred

    def action(self):
        """
        基本的に攻撃。攻撃したい位置が全て攻撃できない座標にある場合のみ移動を行う。
        """

        if self.enemy_w is not None:
            return json.dumps(self.attack(self.enemy_w))
        elif self.enemy_c is not None:
            return json.dumps(self.attack(self.enemy_c))
        elif self.enemy_s is not None:
            return json.dumps(self.attack(self.enemy_s))
        else:
            # 攻撃可能な位置とその点数をリスト型で保持
            attackable_positions = []
            # 攻撃可能な位置を探す
            for i in range(self.field.width):
                for j in range(self.field.height):
                    if self.in_attack_range([i, j]):
                        if self.entire_map[i][j] >= 0:
                            attackable_positions.append(([i, j], self.entire_map[i][j]))
            # 攻撃可能な位置がある場合
            if attackable_positions:
                # 点数が最も高い位置を選ぶ(最大値が複数の場合はランダムに選ぶ)
                best_positions = [pos for pos in attackable_positions if pos[1] == max(attackable_positions, key=lambda x: x[1])[1]]
                print(f"Attackable positions: {best_positions}")
                return json.dumps(self.attack(self.rng.choice(best_positions)[0]))
            else:
                # 攻撃できる位置がない場合は移動
                ship = self.rng.choice(list(self.ships.values()))
                to = self.rng.choice(self.field.squares)
                while not ship.is_reachable(to) or self.overlap(to) is not None or abs(to[0] - ship.position[0]) > 2 or abs(to[1] - ship.position[1]) > 2:
                    # 移動先が到達可能で、重複していない、かつ移動距離が2以内であることを確認
                    to = self.rng.choice(self.field.squares)
                print(f"Moving {ship.type} from {ship.position} to {to}")
                return json.dumps(self.move(ship.type, to))
    
    def update(self, json_str, turn_info):
        super().update(json_str, turn_info)
        c = 0 if turn_info == 'your turn' else 1
        info = self.last_msg
        if c == 0:
            if "result" in info:
                if "moved" in info["result"]:
                    self.predict_position_motion(info["result"]["moved"])
                elif "attacked" in info["result"]:
                    self.predict_position_attack(info["result"]["attacked"])
            self.update_entire_map(info["observation"])
        self.report(info, c)
        print()

    def count_probable_positions(self, prob_map):
        count = 0
        for i in range(self.field.width):
            for j in range(self.field.height):
                if prob_map[i][j] == 1:
                    count += 1
        return count
    
    def report(self, info, c):
        """行動を文章で表示する．"""
        if c == 0:
            player = "you"
        else:
            player = "opponent"
        if "result" in info:
            print(player, end='')
            if "attacked" in info["result"]:
                report_attacked(info["result"]["attacked"])
            elif "moved" in info["result"]:
                report_moved(info["result"]["moved"])

        report_observation(info["observation"])
    
def main(host, part):
    player = OriginalPlayer()
    play_game(host, part, player)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Original Player for Submarine Game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "host",
        help="Hostname of the server, e.g., localhost",
    )
    parser.add_argument(
        "port",
        type=int,
        help="Port of the server, e.g., 2000",
    )
    parser.add_argument(
        "--games", type=int, default=1,
        help="number of games to play (should be consistent with server)",
    )
    args = parser.parse_args()
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    level = logging.INFO
    logging.basicConfig(format=FORMAT, level=level)

    for _ in range(args.games):
        try:
            main(args.host, args.port)
        except KeyboardInterrupt:
            logging.warning('Game interrupted by user')
            break