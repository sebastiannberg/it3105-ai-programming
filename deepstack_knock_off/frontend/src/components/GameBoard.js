import Player from "./Player";

const GameBoard = ({ gameState }) => {
  const playerEntries = gameState.game_players ? Object.entries(gameState.game_players) : [];

  const positions = {
    topPlayers: [],
    leftPlayer: [],
    rightPlayer: [],
    bottomPlayers: []
  };

  // Dynamically distribute players based on the count
  switch(playerEntries.length) {
    case 2:
      positions.topPlayers.push(playerEntries[0]);
      positions.bottomPlayers.push(playerEntries[1]);
      break;
    case 3:
      positions.topPlayers.push(playerEntries[0]);
      positions.topPlayers.push(playerEntries[1]);
      positions.bottomPlayers.push(playerEntries[2]);
      break;
    case 4:
      positions.topPlayers.push(playerEntries[0]);
      positions.rightPlayer.push(playerEntries[1]);
      positions.bottomPlayers.push(playerEntries[2])
      positions.leftPlayer.push(playerEntries[3]);
      break;
    case 5:
      positions.topPlayers.push(playerEntries[0]);
      positions.topPlayers.push(playerEntries[1]);
      positions.rightPlayer.push(playerEntries[2]);
      positions.bottomPlayers.push(playerEntries[3]);
      positions.bottomPlayers.push(playerEntries[4]);
      break;
    case 6:
      positions.topPlayers.push(playerEntries[0]);
      positions.topPlayers.push(playerEntries[1]);
      positions.rightPlayer.push(playerEntries[2]);
      positions.bottomPlayers.push(playerEntries[3]);
      positions.bottomPlayers.push(playerEntries[4]);
      positions.leftPlayer.push(playerEntries[5]);
      break;
    default:
      break;
  }

  const renderPlayers = (playerArray) => playerArray.map(([playerName, playerDetails]) => (
    <Player
      key={playerName}
      name={playerName}
      chips={playerDetails.chips}
      hand={playerDetails.hand}
      bet={playerDetails.player_bet}
    />
  ));

  return (
    <div className="game-board-container">
      <div className="top-players">{renderPlayers(positions.topPlayers, "top-player")}</div>
      <div className="game-board"></div>
      <div className="left-player">{renderPlayers(positions.leftPlayer, "left-player")}</div>
      <div className="right-player">{renderPlayers(positions.rightPlayer, "right-player")}</div>
      <div className="bottom-players">{renderPlayers(positions.bottomPlayers, "bottom-player")}</div>
    </div>
  );
};

export default GameBoard;