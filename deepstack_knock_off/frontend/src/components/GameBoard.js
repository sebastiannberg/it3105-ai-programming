import Card from "./Card";

const GameBoard = ({ gameState }) => {
  console.log(gameState.public_cards)
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
        <div className="public-cards">
          {publicCards}
        </div>
      </div>
  );
};

export default GameBoard;