import { useState, useEffect } from 'react';
import axios from 'axios';
import GameBoard from './GameBoard';
import ActionRow from './ActionRow';


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


  return(
    <div className='board-page'>
      <h1>{gameState.active_player ? Object.keys(gameState.active_player)[0] + "'s turn" : "no current player"}</h1>
      <GameBoard gameState={gameState} />
      <ActionRow />
    </div>
  )
};

export default BoardPage;