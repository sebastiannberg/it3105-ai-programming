import { useState, useEffect } from 'react';
import axios from 'axios';
import GameBoard from './GameBoard';
import ActionPane from './ActionPane';
import Player from './Player';


const BoardPage = () => {
  const [gameState, setGameState] = useState("");
  const [winner, setWinner] = useState(null);
  const [roundWinner, setRoundWinner] = useState(null);

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

  const playerEntries = gameState.game_players ? Object.entries(gameState.game_players) : [];

  const renderPlayers = (playerArray) => playerArray.map(([playerName, playerDetails]) => (
    <Player
      key={playerName}
      name={playerName}
      chips={playerDetails.chips}
      hand={playerDetails.hand}
      bet={playerDetails.player_bet}
    />
  ));

  const handleNextRound = async () => {
    try {
      console.log('Proceeding to the next round...');
      axios.post("http://127.0.0.1:5000/next-round")
        .then(() => {
          fetchGameState()
        })
      setRoundWinner(null);
    } catch (error) {
      console.error('Failed to proceed to the next round:', error);
    }
  };

  return(
    <div className='board-page'>
      <div className='player-section'>
        <h1>Players</h1>
        {renderPlayers(playerEntries)}
      </div>
      <div className='board-section'>
        <h1>
          {winner ? `${winner} is the winner!` :
          roundWinner ? `${roundWinner} has won the round!` :
          gameState.active_player ? `${Object.keys(gameState.active_player)[0]}'s turn` : "no current player"}
        </h1>
        <GameBoard gameState={gameState} />
        <button
          className={`poker-button ${roundWinner ? '' : 'poker-button-invisible'}`}
          onClick={handleNextRound}
          disabled={!roundWinner}
        >
          Next Round
        </button>
      </div>
      <div className='action-section'>
        <h1>Actions</h1>
        <ActionPane gameState={gameState} fetchGameState={fetchGameState} onWinnerDetermined={setWinner} onRoundWinner={setRoundWinner} winner={winner} roundWinner={roundWinner} />
      </div>
    </div>
  )
};

export default BoardPage;