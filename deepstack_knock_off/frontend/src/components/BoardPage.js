import React, { useState, useEffect } from 'react';
import axios from 'axios';
import GameBoard from './GameBoard';
import ActionPane from './ActionPane';
import Player from './Player';

const BoardPage = () => {
  const [gameState, setGameState] = useState(null);
  const [winner, setWinner] = useState(null);
  const [roundWinners, setRoundWinners] = useState([]);
  const [aiMethod, setAiMethod] = useState(null);
  const [aiDecision, setAiDecision] = useState(null);
  const [showAICards, setShowAICards] = useState(false);

  const toggleAICards = () => setShowAICards(!showAICards);

  const fetchGameState = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/game-state");
      setGameState(response.data);
    } catch (error) {
      console.error('Failed to fetch game state:', error);
    }
  };

  useEffect(() => {
    fetchGameState();
  }, []);

  useEffect(() => {
    if (gameState) {
      console.log(gameState)
    }
  }, [gameState])

  const renderPlayers = (playersArray) => {
    return playersArray.map((player, index) => (
      <Player
        key={index}
        name={player.name}
        chips={player.chips}
        hand={player.hand}
        bet={player.player_bet}
        showCards={showAICards || player.name.includes("Human")}
      />
    ));
  };

  const handleNextRound = async () => {
    try {
      console.log('Proceeding to the next round...');
      axios.post("http://127.0.0.1:5000/next-round")
        .then(() => {
          fetchGameState();
        });
      setRoundWinners([]);
    } catch (error) {
      console.error('Failed to proceed to the next round:', error);
    }
  };

  const renderRoundWinnersDetails = () => {
    return roundWinners.map((winner, index) => (
      <div key={index}>
        {winner.hand_category ?
          `${winner.player} won the round with a ${winner.hand_category}` :
          `${winner.player} won the round`}
      </div>
    ));
  };

  return (
    <div className='board-page'>
      <div className='player-section'>
        <h1>Players</h1>
        {gameState ? renderPlayers(gameState.game_players) : <p>Loading players...</p>}
        <button onClick={toggleAICards} className="poker-button">
          {showAICards ? 'Hide AI Cards' : 'Show AI Cards'}
        </button>
      </div>
      <div className='board-section'>
        <h1>
          {winner ? `${winner} is the winner!` :
          roundWinners.length > 0 ? renderRoundWinnersDetails() :
          gameState?.current_player ? `${gameState.current_player.name}'s turn` : "No current player"}
        </h1>
        {
          (winner || roundWinners.length > 0) ? <p>&nbsp;</p> :
          (gameState?.current_player && gameState.current_player.name.includes("AI")) ?
            (aiDecision && aiDecision.action_name ? <p>AI chose to <strong>{aiDecision.action_name}</strong> using <strong>{aiMethod}</strong></p> : <p>AI calculating decision using <strong>{aiMethod}</strong>...</p>)
            : <p>&nbsp;</p>
        }
        <GameBoard gameState={gameState} />
        <button
          className={`poker-button ${roundWinners.length > 0 ? '' : 'poker-button-invisible'}`}
          onClick={handleNextRound}
          disabled={roundWinners.length === 0}
        >
          Next Round
        </button>
      </div>
      <div className='action-section'>
        <h1>Actions</h1>
        <ActionPane gameState={gameState} fetchGameState={fetchGameState} onWinnerDetermined={setWinner} onRoundWinners={setRoundWinners} onAiDecision={setAiDecision} setAiMethod={setAiMethod} winner={winner} roundWinners={roundWinners} aiDecision={aiDecision} />
      </div>
    </div>
  );
};

export default BoardPage;
