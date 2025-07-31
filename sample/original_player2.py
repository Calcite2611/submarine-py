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
        msg += " hit " + str(attacked["hit"]) # str()で囲んで安全にする
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
        self.enemy_w = None
        self.enemy_c = None
        self.enemy_s = None
        self.w_pos_pred = None
        self.c_pos_pred = None
        self.s_pos_pred = None
        self.entire_map = None

    def initialize(self, field):
        super().initialize(field)
        # 予測マップを0（未確定）で初期化
        self.w_pos_pred = np.zeros((self.field.width, self.field.height))
        self.c_pos_pred = np.zeros((self.field.width, self.field.height))
        self.s_pos_pred = np.zeros((self.field.width, self.field.height))
        self.entire_map = np.zeros((self.field.width, self.field.height))

    def name(self):
        return 'original-player'
    
    def place_ship(self):
        distance = 2
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
            if not placed:
                for r in range(self.field.height):
                    for c in range(self.field.width):
                        if (c, r) not in placed_positions:
                            placed_positions.add((c, r))
                            placed = True
                            break
                    if placed:
                        break
        placed_list = list(placed_positions)
        return {
            'w': list(placed_list[0]),
            'c': list(placed_list[1]),
            's': list(placed_list[2])
        }
    
    ### 修正点 4 ###
    def predict_position_motion(self, moved):
        """相手の行動が移動であった場合の予測マップの更新を行う"""
        ship_type = moved['ship']
        distance = moved['distance']

        if ship_type == 'w':
            if self.enemy_w is not None:
                self.enemy_w[0] += distance[0]
                self.enemy_w[1] += distance[1]
            # 常に予測マップも更新する
            self.w_pos_pred = self.update_prediction_map_motion(self.w_pos_pred, distance)
        elif ship_type == 'c':
            if self.enemy_c is not None:
                self.enemy_c[0] += distance[0]
                self.enemy_c[1] += distance[1]
            self.c_pos_pred = self.update_prediction_map_motion(self.c_pos_pred, distance)
        elif ship_type == 's':
            if self.enemy_s is not None:
                self.enemy_s[0] += distance[0]
                self.enemy_s[1] += distance[1]
            self.s_pos_pred = self.update_prediction_map_motion(self.s_pos_pred, distance)
    
    def update_prediction_map_motion(self, target_map, motion):
        dx, dy = motion
        new_pred_map = np.full_like(target_map, notExist)
        for y in range(self.field.height):
            for x in range(self.field.width):
                if target_map[x, y] != notExist:
                    nx, ny = x - dx, y - dy # 元の位置を計算
                    if 0 <= nx < self.field.width and 0 <= ny < self.field.height:
                        new_pred_map[x, y] = target_map[nx, ny]
        return new_pred_map
    
    ### 修正点 2 & 3 ###
    def predict_position_attack(self, attacked):
        """相手の行動が攻撃であった場合の予測マップの更新を行う（アルゴリズムに沿って全面改修）"""
        pos = attacked['position']
        x, y = pos

        # 攻撃が確定位置と一致していたら、その確定情報をリセット
        if self.enemy_w == pos: self.enemy_w = None
        if self.enemy_c == pos: self.enemy_c = None
        if self.enemy_s == pos: self.enemy_s = None

        # ヒットした場合
        if 'hit' in attacked:
            hit_ship_type = attacked['hit']
            if not isinstance(hit_ship_type, str): hit_ship_type = hit_ship_type[0]

            target_map = None
            if hit_ship_type == 'w':
                self.enemy_w = pos
                target_map = self.w_pos_pred
            elif hit_ship_type == 'c':
                self.enemy_c = pos
                target_map = self.c_pos_pred
            elif hit_ship_type == 's':
                self.enemy_s = pos
                target_map = self.s_pos_pred
            
            # ヒットした艦のマップを更新：ヒット地点を1、他をnotExistに
            if target_map is not None:
                target_map.fill(notExist)
                target_map[x, y] = 1

        # ニアミスした場合
        if 'near' in attacked:
            near_ship_types = attacked['near']
            for ship_type in near_ship_types:
                target_map = None
                if ship_type == 'w': target_map = self.w_pos_pred
                elif ship_type == 'c': target_map = self.c_pos_pred
                elif ship_type == 's': target_map = self.s_pos_pred

                # ニアミスした艦のマップを更新：中心をnotExist、周囲8マスを1、他をnotExistに
                if target_map is not None:
                    target_map.fill(notExist)
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if i == 0 and j == 0: continue
                            px, py = x + i, y + j
                            if 0 <= px < self.field.width and 0 <= py < self.field.height:
                                target_map[px, py] = 1

        # 完全なミスだった場合
        if 'hit' not in attacked and 'near' not in attacked:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    px, py = x + i, y + j
                    if 0 <= px < self.field.width and 0 <= py < self.field.height:
                        self.w_pos_pred[px, py] = notExist
                        self.c_pos_pred[px, py] = notExist
                        self.s_pos_pred[px, py] = notExist

    def update_entire_map(self, observation):
        """全体の予測マップを更新する"""
        self.entire_map.fill(0) # 0でリセット
        for ship_type, state in observation['opponent'].items():
            if state["hp"] > 0:
                if ship_type == 'w': self.entire_map += self.w_pos_pred
                elif ship_type == 'c': self.entire_map += self.c_pos_pred
                elif ship_type == 's': self.entire_map += self.s_pos_pred
    
    ### 修正点 1 ###

    def action(self):
        """行動決定ロジック（エラー箇所を修正）"""
        # 優先攻撃対象（確定位置）がいれば攻撃
        if self.enemy_w is not None and self.in_attack_range(self.enemy_w):
            return json.dumps(self.attack(self.enemy_w))
        if self.enemy_c is not None and self.in_attack_range(self.enemy_c):
            return json.dumps(self.attack(self.enemy_c))
        if self.enemy_s is not None and self.in_attack_range(self.enemy_s):
            return json.dumps(self.attack(self.enemy_s))

        # 予測マップに基づいて攻撃
        attackable_positions = []
        for r in range(self.field.height):
            for c in range(self.field.width):
                pos = [c, r]
                # 攻撃可能で、かつ存在可能性が「なし」ではないマスを候補に
                if self.in_attack_range(pos) and self.entire_map[c, r] > notExist:
                    attackable_positions.append((pos, self.entire_map[c, r]))

        if attackable_positions:
            # スコアが最も高い位置を攻撃（複数あればランダム）
            attackable_positions.sort(key=lambda x: x[1], reverse=True)
            best_score = attackable_positions[0][1]
            best_positions = [pos for pos, score in attackable_positions if score == best_score]
            return json.dumps(self.attack(self.rng.choice(best_positions)))
        else:
            # 攻撃できる位置がない場合は移動
            # ### 修正点 ### 'is_movable'は存在しないため削除
            movable_ships = list(self.ships.values())
            if not movable_ships: return json.dumps(self.attack([0, 0])) # 緊急避難

            ship_to_move = self.rng.choice(movable_ships)
            to = self.rng.choice(self.field.squares)
            attempts = 0
            while (not ship_to_move.is_reachable(to) or self.overlap(to) is not None) and attempts < 100:
                to = self.rng.choice(self.field.squares)
                attempts += 1
            return json.dumps(self.move(ship_to_move.type, to))

    def update(self, json_str, turn_info):
        super().update(json_str, turn_info)
        c = 0 if turn_info == 'your turn' else 1
        info = self.last_msg

        # ### 修正案（参考）###
        # 相手の攻撃情報も利用するとより強くなる
        # if "result" in info and "attacked" in info["result"]:
        #    # ここで相手の攻撃情報から予測マップを更新するロジックを追加
        #    pass
        
        if c == 0: # 自分のターンの結果
            if "result" in info and "attacked" in info["result"]:
                self.predict_position_attack(info["result"]["attacked"])
        else: # 相手のターンの結果
             if "result" in info and "moved" in info["result"]:
                self.predict_position_motion(info["result"]["moved"])

        # 常に最新の観測情報で全体マップを更新
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
    parser.add_argument("host", help="Hostname of the server")
    parser.add_argument("port", type=int, help="Port of the server")
    parser.add_argument("--games", type=int, default=1, help="number of games to play")
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
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            break