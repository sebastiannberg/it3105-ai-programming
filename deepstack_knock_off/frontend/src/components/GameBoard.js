import Player from "./Player";
import Card from "./Card";

const GameBoard = ({ gameState }) => {
  const publicCards = gameState.public_cards ? (
    gameState.public_cards.map((card, index) => (
      <Card key={index} rank={card.rank} suit={card.suit} />
    ))
  ) : null;

  return (
      <div className="game-board">
        <div>
          <h4>Total Pot: {gameState.pot}</h4>
          <h5>Current Bet: {gameState.current_bet}</h5>
        </div>
        <p>{publicCards}</p>
      </div>
  );
};

export default GameBoard;