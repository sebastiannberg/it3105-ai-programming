import { useState, useEffect } from 'react';
import axios from 'axios';


const BoardPage = () => {
  const [gameState, setGameState] = useState("");

  const fetchGameState = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/game-state");
      setGameState(JSON.stringify(response.data));
    } catch (error) {
      console.error('Failed to fetch game state:', error);
    }
  };

  useEffect(() => {
    fetchGameState()
  }, []);


  return(
    <div>
      <h1 className='title'>BoardPage</h1>
      <div className='game-board'>
        <p>{gameState}</p>
      </div>
    </div>
  )
};

export default BoardPage;