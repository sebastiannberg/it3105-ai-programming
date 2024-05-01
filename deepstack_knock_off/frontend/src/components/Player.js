import Card from "./Card";


const Player = ({ name, chips, hand, bet, showCards }) => {
  const handCards = showCards ? hand.map((card, index) => (
    <Card key={index} rank={card.rank} suit={card.suit} />
  )) : null;

  return (
    <div className="player">
      <div className="player-info">
        <h3>{name}</h3>
        <p>Chips: {chips}</p>
        <p>Current Bet: {bet}</p>
      </div>
      {handCards}
    </div>
  )
};

export default Player;