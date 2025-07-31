from submarine_py import Player, play_game, Ship, Reporter, Field
import json
import random
import logging
import numpy as np

notExist = -0.25  # 予測マップ上で存在しない位置を表す値

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
        # ### 修正点 ### 全てのマスが等しい確率を持つように1で初期化
        self.w_pos_pred = np.ones((self.field.width, self.field.height))
        self.c_pos_pred = np.ones((self.field.width, self.field.height))
        self.s_pos_pred = np.ones((self.field.width, self.field.height))
        self.entire_map = np.zeros((self.field.width, self.field.height))

    def name(self):
        return 'original-player'
    
    def place_ship(self):
        distance = int(min(self.field.height, self.field.width) // 2)
        placed_positions = set()
        ship_types = ['w', 'c', 's']
        max_attempts = 500
        for ship_type in ship_types:
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                attempts += 1
                row = self.rng.randint(0, self.field.height - 1)
                col = self.rng.randint(0, self.field.width - 1)
                position = [col, row]

                if tuple(position) not in placed_positions:
                    is_isolated = True
                    for placed_pos in placed_positions:
                        if abs(position[1] - placed_pos[1]) <= distance and abs(position[0] - placed_pos[0]) <= distance:
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
                            placed = True
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
            # ### 修正点 ### 位置確定時も予測マップを更新する
            self.w_pos_pred = self.update_prediction_map_motion(self.w_pos_pred, moved['distance'])
        elif moved['ship'] == 'c':
            if self.enemy_c is not None:
                self.enemy_c[0] += moved['distance'][0]
                self.enemy_c[1] += moved['distance'][1]
            self.c_pos_pred = self.update_prediction_map_motion(self.c_pos_pred, moved['distance'])
        elif moved['ship'] == 's':
            if self.enemy_s is not None:
                self.enemy_s[0] += moved['distance'][0]
                self.enemy_s[1] += moved['distance'][1]
            self.s_pos_pred = self.update_prediction_map_motion(self.s_pos_pred, moved['distance'])
    
    def update_prediction_map_motion(self, target_map, motion):
        dx, dy = motion
        # 新しいマップをnotExistで初期化
        new_pred_map = np.full_like(target_map, notExist)
        # 元のマップの各セルの値を移動後の位置にコピー
        for y in range(self.field.height):
            for x in range(self.field.width):
                # 移動元の値がnotExistでなければ移動させる
                if target_map[x, y] != notExist:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.field.width and 0 <= ny < self.field.height:
                        new_pred_map[nx, ny] = target_map[x, y]
        return new_pred_map
    
    def predict_position_attack(self, attacked):
        """
        相手の行動が攻撃であった場合の予測マップの更新を行う
        """
        x, y = attacked['position']

        ## ### 修正点 1 ### 攻撃が当たった場合の処理を全面的に修正
        if 'hit' in attacked:
            hit_ship_type = attacked['hit']
            hit_pos = attacked['position']
            
            if hit_ship_type == 'w':
                self.enemy_w = hit_pos
                self.w_pos_pred = np.full_like(self.w_pos_pred, notExist)
                self.w_pos_pred[hit_pos[0], hit_pos[1]] = 1
            elif hit_ship_type == 'c':
                self.enemy_c = hit_pos
                self.c_pos_pred = np.full_like(self.c_pos_pred, notExist)
                self.c_pos_pred[hit_pos[0], hit_pos[1]] = 1
            elif hit_ship_type == 's':
                self.enemy_s = hit_pos
                self.s_pos_pred = np.full_like(self.s_pos_pred, notExist)
                self.s_pos_pred[hit_pos[0], hit_pos[1]] = 1

        ## ### 修正点 3 ### 水飛沫（ニアミス）が発生した場合の処理を修正
        if 'near' in attacked:
            near_ship_type = attacked['near']
            target_map = None
            if near_ship_type == 'w': target_map = self.w_pos_pred
            elif near_ship_type == 'c': target_map = self.c_pos_pred
            elif near_ship_type == 's': target_map = self.s_pos_pred

            if target_map is not None:
                # 攻撃中心点は存在しない
                target_map[x, y] = notExist
                # 周囲8マスのスコアを加算
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if i == 0 and j == 0: continue
                        px, py = x + i, y + j
                        if 0 <= px < self.field.width and 0 <= py < self.field.height:
                            if target_map[px, py] != notExist:
                                target_map[px, py] += 1  # 上書きではなく加算

        ## 何も当たらなかった場合の処理
        if 'hit' not in attacked and 'near' not in attacked:
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    px, py = x + i, y + j
                    if 0 <= px < self.field.width and 0 <= py < self.field.height:
                        self.w_pos_pred[px, py] = notExist
                        self.c_pos_pred[px, py] = notExist
                        self.s_pos_pred[px, py] = notExist

    def update_entire_map(self, observation):
        """
        全体の予測マップを更新する
        """
        ### 修正点 2 ### 撃沈された艦の位置確定情報をリセット
        for ship_type, state in observation['opponent'].items():
            if state["hp"] <= 0:
                if ship_type == 'w': self.enemy_w = None
                elif ship_type == 'c': self.enemy_c = None
                elif ship_type == 's': self.enemy_s = None

        self.entire_map = np.zeros((self.field.width, self.field.height))
        for ship_type, state in observation['opponent'].items():
            if state["hp"] > 0:
                if ship_type == 'w':
                    self.entire_map += self.w_pos_pred
                elif ship_type == 'c':
                    self.entire_map += self.c_pos_pred
                elif ship_type == 's':
                    self.entire_map += self.s_pos_pred

    def action(self):
        """
        基本的に攻撃。攻撃したい位置が全て攻撃できない座標にある場合のみ移動を行う。
        """
        # 優先順位に従って攻撃
        if self.enemy_w is not None and self.in_attack_range(self.enemy_w):
            return json.dumps(self.attack(self.enemy_w))
        if self.enemy_c is not None and self.in_attack_range(self.enemy_c):
            return json.dumps(self.attack(self.enemy_c))
        if self.enemy_s is not None and self.in_attack_range(self.enemy_s):
            return json.dumps(self.attack(self.enemy_s))
        
        # 攻撃可能な位置とその点数をリストで保持
        attackable_positions = []
        for r in range(self.field.height):
            for c in range(self.field.width):
                pos = [c, r]
                # ### 修正点 ### in_attack_rangeのチェックを追加
                if self.in_attack_range(pos) and self.entire_map[c, r] > notExist:
                    attackable_positions.append((pos, self.entire_map[c, r]))
        
        # 点数が最も高い位置を攻撃
        if attackable_positions:
            attackable_positions.sort(key=lambda x: x[1], reverse=True)
            best_score = attackable_positions[0][1]
            best_positions = [pos for pos, score in attackable_positions if score == best_score]
            target_pos = self.rng.choice(best_positions)
            return json.dumps(self.attack(target_pos))
        else:
            # 攻撃できる位置がない場合は移動
            # 移動可能な艦をリストアップ
            movable_ships = [ship for ship in self.ships.values() if ship.is_movable()]
            if not movable_ships: # 動ける艦がない場合（ありえないが念のため）
                return json.dumps(self.attack([0,0])) # ダミー攻撃

            ship_to_move = self.rng.choice(movable_ships)
            
            # 移動先を探す
            possible_moves = []
            for r in range(self.field.height):
                for c in range(self.field.width):
                    to = [c, r]
                    if ship_to_move.is_reachable(to) and self.overlap(to) is None:
                         # 移動コスト(ターン数)が1の範囲に限定するなどの戦略も可能
                         if abs(to[0] - ship_to_move.position[0]) + abs(to[1] - ship_to_move.position[1]) <= 2:
                            possible_moves.append(to)
            
            if possible_moves:
                to = self.rng.choice(possible_moves)
                return json.dumps(self.move(ship_to_move.type, to))
            else: # 移動先もない場合
                return json.dumps(self.attack([0,0])) # ダミー攻撃

    
    def update(self, json_str, turn_info):
        super().update(json_str, turn_info)
        c = 0 if turn_info == 'your turn' else 1
        info = self.last_msg
        if c == 0: # 自分のターン結果の時のみ予測を更新
            if "result" in info:
                if "moved" in info["result"]:
                    # 相手が移動した結果はこちらには通知されないので、このパスは通らないはず
                    pass
                elif "attacked" in info["result"]:
                    self.predict_position_attack(info["result"]["attacked"])
            self.update_entire_map(info["observation"])
        else: # 相手のターンの時
            if "result" in info and "moved" in info["result"]:
                 self.predict_position_motion(info["result"]["moved"])
                 self.update_entire_map(info["observation"])

        self.report(info, c)
        print()
    
    def report(self, info, c):
        """行動を文章で表示する．"""
        player = "you" if c == 0 else "opponent"
        if "result" in info:
            print(player, end='')
            if "attacked" in info["result"]:
                report_attacked(info["result"]["attacked"])
            elif "moved" in info["result"]:
                report_moved(info["result"]["moved"])
        report_observation(info["observation"])

# main関数は変更なし
def main(host, port):
    player = OriginalPlayer()
    play_game(host, port, player)

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