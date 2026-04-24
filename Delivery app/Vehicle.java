abstract class Vehicle implements Delivery {
    @Override
    public void delivered() {
        System.out.println("the item has been delivered");
    }
}