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
        # str()で囲むことで、中身がリストでも文字列でも安全に表示できる
        msg += " hit " + str(attacked["hit"])
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
        self.rng = random.Random(42)
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
        # フィールドが設定された後に予測マップを初期化(全て0に設定)
        self.w_pos_pred = np.zeros((self.field.width, self.field.height))
        self.c_pos_pred = np.zeros((self.field.width, self.field.height))
        self.s_pos_pred = np.zeros((self.field.width, self.field.height))
        self.entire_map = np.zeros((self.field.width, self.field.height))

    def name(self):
        return 'original-player'
    
    def place_ship(self):
        distance = 2
        placements = {}  # 最終的な配置結果を格納する「辞書」
        occupied_coords = set()  # 占有済みの座標を記録する「セット」
        
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

                if tuple(position) not in occupied_coords:
                    is_isolated = True
                    for placed_pos in occupied_coords:
                        if abs(position[1] - placed_pos[1]) <= distance and abs(position[0] - placed_pos[0]) <= distance:
                            is_isolated = False
                            break
                    
                    if is_isolated:
                        placements[ship_type] = position      # 辞書に結果を保存
                        occupied_coords.add(tuple(position))  # セットに座標を追加
                        placed = True
                        logging.info(f"Placed {ship_type} at {position}")
            
            if not placed:
                logging.warning(f"Failed to place {ship_type} after {max_attempts} attempts. Trying to place in a random way.")
                for r in range(self.field.height):
                    for c in range(self.field.width):
                        if (c, r) not in occupied_coords:
                            position = [c, r]
                            placements[ship_type] = position     # 辞書に結果を保存
                            occupied_coords.add(tuple(position)) # セットに座標を追加
                            logging.info(f"Placed {ship_type} at {position}")
                            placed = True
                            break
                    if placed:
                        break

        return placements
    
    def predict_position_motion(self, moved):
        """
        相手の行動が移動であった場合の予測マップの更新を行う
        """
        if moved['ship'] == 'w':
            if self.enemy_w is not None:
                self.enemy_w[0] += moved['distance'][0]
                self.enemy_w[1] += moved['distance'][1]
            self.w_pos_pred = self.slide_map(self.w_pos_pred, moved['distance'])
        elif moved['ship'] == 'c':
            if self.enemy_c is not None:
                self.enemy_c[0] += moved['distance'][0]
                self.enemy_c[1] += moved['distance'][1]
            self.c_pos_pred = self.slide_map(self.c_pos_pred, moved['distance'])
        elif moved['ship'] == 's':
            if self.enemy_s is not None:
                self.enemy_s[0] += moved['distance'][0]
                self.enemy_s[1] += moved['distance'][1]
            self.s_pos_pred = self.slide_map(self.s_pos_pred, moved['distance'])

    def slide_map(self, map, motion):
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
        # 確定情報をリセット
        pos = attacked['position']
        if self.enemy_w == pos: self.enemy_w = None
        if self.enemy_c == pos: self.enemy_c = None
        if self.enemy_s == pos: self.enemy_s = None

        x = attacked['position'][0]
        y = attacked['position'][1]

        ## 攻撃が当たった場合の処理
        if 'hit' in attacked:
            self.onHit(attacked['hit'], attacked['position'])

        ## 水飛沫が発生した場合の処理
        if 'near' in attacked:
            self.onNear(attacked['near'], attacked['position'])

        ## 何も当たらなかった場合の処理
        if 'hit' not in attacked and 'near' not in attacked:
            self.noInfo_by_attack(attacked['position'])

    def onHit(self, ship_type, position):
        """
        攻撃が当たった場合の処理
        """
        if ship_type == 'w':
            self.enemy_w = position
            self.w_pos_pred = np.full((self.field.width, self.field.height), notExist)
            self.w_pos_pred[position[0]][position[1]] = 1
        elif ship_type == 'c':
            self.enemy_c = position
            self.c_pos_pred = np.full((self.field.width, self.field.height), notExist)
            self.c_pos_pred[position[0]][position[1]] = 1
        elif ship_type == 's':
            self.enemy_s = position
            self.s_pos_pred = np.full((self.field.width, self.field.height), notExist)
            self.s_pos_pred[position[0]][position[1]] = 1

    def onNear(self, ship_types, position):
        """
        水飛沫が発生した場合の処理（merge_mapsを使用）
        """
        # 1. 今回のニアミス情報だけを持つ一時的なマップを作成
        near_info_map = np.zeros((self.field.width, self.field.height))
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0: continue
                px, py = position[0] + i, position[1] + j
                if 0 <= px < self.field.width and 0 <= py < self.field.height:
                    near_info_map[px, py] = 1 # 可能性のある8マスを1にする

        # 2. 該当する艦の既存マップと、今作った一時マップを統合する
        for ship_type in ship_types:
            if ship_type == 'w':
                self.w_pos_pred = self.merge_maps(self.w_pos_pred, near_info_map)
            elif ship_type == 'c':
                self.c_pos_pred = self.merge_maps(self.c_pos_pred, near_info_map)
            elif ship_type == 's':
                self.s_pos_pred = self.merge_maps(self.s_pos_pred, near_info_map)
        
    def noInfo_by_attack(self, position):
        """
        攻撃が当たらなかった場合の処理
        """
        # ミスマップの作成
        miss_info_map = np.zeros((self.field.width, self.field.height))
        
        # 周囲9マスをnotExistに設定
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                px, py = position[0] + i, position[1] + j
                if 0 <= px < self.field.width and 0 <= py < self.field.height:
                    miss_info_map[px, py] = notExist

        # 新しいミスマップを全ての予測マップに統合
        self.w_pos_pred = self.merge_maps(self.w_pos_pred, miss_info_map)
        self.c_pos_pred = self.merge_maps(self.c_pos_pred, miss_info_map)
        self.s_pos_pred = self.merge_maps(self.s_pos_pred, miss_info_map)

    def merge_maps(self, map1, map2):
        """
        2つの予測マップを合成する。
        両方のマップで可能性が否定されていない（notExistでない）場合のみ、情報を引き継ぐ。
        """ 
        # 結果を格納するための新しいマップを準備
        merged_map = np.full_like(map1, notExist)

        for r in range(self.field.height):
            for c in range(self.field.width):
                # 両方のマップで可能性が否定されていない（notExistでない）場合のみ、情報を引き継ぐ
                is_possible_in_map1 = map1[c, r] > notExist
                is_possible_in_map2 = map2[c, r] > notExist

                if is_possible_in_map1 and is_possible_in_map2:
                    # スコアは単純に足し合わせることで、情報の重みを表現できる
                    merged_map[c, r] = map1[c, r] + map2[c, r]
        
        return merged_map

    def update_entire_map(self, observation):
        """
        全体の予測マップを更新する。
        """
        # 敵艦が沈んだ場合は、対応する位置をNoneに設定
        for ship_type, state in observation['opponent'].items():
            if state["hp"] <= 0:
                if ship_type == 'w' and self.enemy_w is not None:
                    self.enemy_w = None
                    logging.info("Opponent's w has been sunk. Resetting enemy_w.")
                elif ship_type == 'c' and self.enemy_c is not None:
                    self.enemy_c = None
                    logging.info("Opponent's c has been sunk. Resetting enemy_c.")
                elif ship_type == 's' and self.enemy_s is not None:
                    self.enemy_s = None
                    logging.info("Opponent's s has been sunk. Resetting enemy_s.")

        # 全体マップを更新
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
        行動決定ロジックを修正。探索的攻撃を追加。
        """
        # 1. 最優先：確定位置の敵が攻撃範囲内にいれば攻撃
        if self.enemy_w is not None and self.in_attack_range(self.enemy_w):
            return json.dumps(self.attack(self.enemy_w))
        if self.enemy_c is not None and self.in_attack_range(self.enemy_c):
            return json.dumps(self.attack(self.enemy_c))
        if self.enemy_s is not None and self.in_attack_range(self.enemy_s):
            return json.dumps(self.attack(self.enemy_s))

        # 2. 次善策：スコアがプラスの「可能性の高い」マスを探す
        good_positions = []
        for r in range(self.field.height):
            for c in range(self.field.width):
                pos = [c, r]
                if self.in_attack_range(pos) and self.entire_map[c, r] > 0:
                    good_positions.append((pos, self.entire_map[c, r]))

        if good_positions:
            best_score = max(pos[1] for pos in good_positions)
            best_positions = [pos for pos, score in good_positions if score == best_score]
            target_pos = self.rng.choice(best_positions)
            return json.dumps(self.attack(target_pos))

        # 3. 探索的攻撃：スコアが0以上の「まだ可能性のある」マスを探す
        possible_positions = []
        for r in range(self.field.height):
            for c in range(self.field.width):
                pos = [c, r]
                if self.in_attack_range(pos) and self.entire_map[c, r] > notExist:
                    possible_positions.append(pos)
        
        if possible_positions:
            # 可能性のあるマスからランダムに選んで攻撃
            target_pos = self.rng.choice(possible_positions)
            return json.dumps(self.attack(target_pos))

        # 4. 最終手段：攻撃できるマスがなければ移動
        movable_ships = list(self.ships.values())
        if not movable_ships: return json.dumps(self.attack([0,0]))

        ship_to_move = self.rng.choice(movable_ships)
        
        move_candidates = []
        for to in self.field.squares:
            if ship_to_move.is_reachable(to) and self.overlap(to) is None:
                 move_candidates.append(to)
        
        if move_candidates:
            to = self.rng.choice(move_candidates)
            print(f"Moving {ship_to_move.type} from {ship_to_move.position} to {to}")
            return json.dumps(self.move(ship_to_move.type, to))
        else:
            # 移動もできない場合の最終手段
            return json.dumps(self.attack([0,0]))
        
    def update(self, json_str, turn_info):
        super().update(json_str, turn_info)
        info = self.last_msg
        c = 0 if turn_info == 'your turn' else 1

        if "result" in info:
            # "moved"は常に相手の行動なので、いつでも学習する
            if "moved" in info["result"]:
                self.predict_position_motion(info["result"]["moved"])
            
            # "attacked"はターンによって意味が違うので、処理を分ける
            elif "attacked" in info["result"]:
                if c == 0: # あなたのターンの場合：敵への攻撃結果
                    self.predict_position_attack(info["result"]["attacked"])
                else: # 相手のターンの場合：あなたへの攻撃結果
                    # ここで相手の攻撃から位置を推測するロジックを追加できると、さらに強くなります。
                    # しかし、まずは間違った学習を防ぐために、何もしない(pass)のが安全です。
                    pass

        # 常に最新の観測情報で全体マップを更新する
        self.update_entire_map(info["observation"])
        
        # レポート表示
        self.report(info, c)
        print()
    
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