import { useState, useEffect } from 'react';
import axios from 'axios';
import GameBoard from './GameBoard';
import ActionPane from './ActionPane';
import Player from './Player';


const BoardPage = () => {
  const [gameState, setGameState] = useState("");

  const fetchGameState = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/game-state");
      setGameState(response.data);
    } catch (error) {
      console.error('Failed to fetch game state:', error);
    }
  };

  useEffect(() => {
    fetchGameState()
  }, []);

  const playerEntries = gameState.round_players ? Object.entries(gameState.round_players) : [];

  const renderPlayers = (playerArray) => playerArray.map(([playerName, playerDetails]) => (
    <Player
      key={playerName}
      name={playerName}
      chips={playerDetails.chips}
      hand={playerDetails.hand}
      bet={playerDetails.player_bet}
    />
  ));

  return(
    <div className='board-page'>
      <div className='player-section'>
        <h1>Players</h1>
        {renderPlayers(playerEntries)}
      </div>
      <div className='board-section'>
        <h1>{gameState.active_player ? Object.keys(gameState.active_player)[0] + "'s turn" : "no current player"}</h1>
        <GameBoard gameState={gameState} />
      </div>
      <div className='action-section'>
        <h1>Actions</h1>
        <ActionPane gameState={gameState} />
      </div>
    </div>
  )
};

export default BoardPage;