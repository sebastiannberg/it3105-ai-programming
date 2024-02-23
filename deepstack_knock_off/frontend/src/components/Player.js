import Card from "./Card";


const Player = ({ name, chips, hand, bet }) => {
  const handCards = hand.map((card, index) => (
    <Card key={index} rank={card.rank} suit={card.suit} />
  ));

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